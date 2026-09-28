"""Build a TensorRT engine bundle for serve.py from a Laya checkpoint.

    uv run --group build build_engine.py                                  # -> engines/laya-fp16/
    uv run --group build build_engine.py --dtype bf16
    uv run --group build build_engine.py --max-batch 128 --opt-batch 32 --output engines/big

Three stages:
  1. Export: load the PyTorch checkpoint with `laya` and export its network to ONNX, in the
     requested precision. (TensorRT 11 networks are strongly typed: the engine runs in exactly the
     dtypes of the ONNX graph, so precision is decided here, not at build time.)
  2. Build: compile the ONNX graph into a TensorRT engine for *this* GPU and TensorRT version.
  3. Bundle: write model.engine next to the checkpoint's tokenizer and calibration config, plus a
     build.json describing how it was built. serve.py needs nothing else.

Then it verifies the bundle: the same questions answered through serve.py's pipeline and through
stock Laya in PyTorch must reach the same decisions.

This script needs the `build` dependency group (laya, onnx, onnxscript); serve.py does not.
An engine only runs on the GPU architecture and TensorRT version that built it, so build it on
(or for) the machine that will serve it.
"""
import argparse
import gc
import json
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt
import torch
from torch.export import Dim

import serve

HERE = Path(__file__).resolve().parent
OUTPUT_NAMES = ["logits", "act_logits"]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
# Each engine input's axes, in order. The ONNX export and the optimization profile both use this.
AXES = {"input_ids": ("batch", "seq"), "attention_mask": ("batch", "seq"),
        "marker_pos": ("batch", "options"), "marker_mask": ("batch", "options"), "qtype": ("batch",)}


def load_laya(checkpoint: str, device: str):
    import laya
    with warnings.catch_warnings():
        # The shipped checkpoint has one out-of-range calibration temperature; laya warns on every
        # load. It affects confidence numbers only (and serve.py clamps it the same way).
        warnings.simplefilter("ignore", RuntimeWarning)
        return laya.load(checkpoint, device=device)


class CastInput(torch.nn.Module):
    """Run `inner` in its own weight dtype and return fp32.

    DecisionModel.forward() feeds `h[:, 0].float()` (fp32) into `act_head` (Linear layers). Laya
    always runs under torch.autocast, which silently casts it back down; without autocast a bf16 or
    fp16 model has mismatched matmul inputs, which TensorRT's strongly typed parser rejects. This
    does explicitly what autocast did implicitly.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.dtype = next(inner.parameters()).dtype

    def forward(self, x):
        return self.inner(x.to(self.dtype)).float()


def export_onnx(checkpoint: str, dtype: str, onnx_path: Path) -> None:
    agent = load_laya(checkpoint, "cpu")   # tracing needs no GPU
    model = agent.model.eval().to(DTYPES[dtype])
    if dtype != "fp32":
        model.act_head = CastInput(model.act_head)
    # Example inputs: values don't matter, shapes/dtypes do. 3 rows x 16 tokens x 3 options.
    # NOTE: forward() has a Python branch for questions with a single option. Tracing records only
    # the branch the example takes (>= 2 options), which is why the profile's minimum is 2 options.
    example = (torch.randint(0, 1000, (3, 16)), torch.ones(3, 16, dtype=torch.long),
               torch.tensor([[1, 5, 9]] * 3), torch.ones(3, 3, dtype=torch.bool), torch.tensor([0, 1, 2]))
    dims = {d: Dim(d) for d in ("batch", "seq", "options")}
    dynamic_shapes = {name: {i: dims[d] for i, d in enumerate(AXES[name])} for name in serve.INPUT_NAMES}
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, str(onnx_path), input_names=serve.INPUT_NAMES, output_names=OUTPUT_NAMES,
                      dynamic_shapes=dynamic_shapes, dynamo=True,
                      external_data=True)   # weights in a side file: protobuf caps a file at 2 GiB
    del agent, model
    gc.collect()


def build_engine(onnx_path: Path, engine_path: Path, profile_dims: dict, tf32: bool, workspace_gb: int,
                 verbose: bool) -> None:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):   # NOTE: use from_file, so it finds the external weights, we're close to the 2GB onnx protobuf filesize 
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"TensorRT could not parse {onnx_path}:\n{errors}")

    # Builder config: workspace memory limit and TF32 setting.
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)

    # Optimization profile: the (min, opt, max) shape range each input's dynamic axes may take.
    profile = builder.create_optimization_profile()
    for name, axes in AXES.items():
        profile.set_shape(name, *(tuple(profile_dims[d][i] for d in axes) for i in range(3)))
    config.add_optimization_profile(profile)

    # The timing cache remembers kernel timings, so rebuilding on the same GPU is much faster.
    cache_path = HERE / "build" / "timing.cache"
    cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    # Compile the engine, write it out, and persist the updated timing cache.
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("TensorRT build failed; rerun with --verbose to see why")
    engine_path.write_bytes(plan)
    cache_path.write_bytes(config.get_timing_cache().serialize())


def verify(bundle: Path, checkpoint: str) -> bool:
    """Answer sample questions through serve.py's pipeline and through stock Laya; compare decisions."""
    # Our side: the TensorRT engine behind serve.py's pipeline.
    decider = serve.Decider(bundle / "model.engine", bundle)
    # Reference side: the stock PyTorch Laya model on GPU.
    reference = load_laya(checkpoint, "cuda")

    states = [
        "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
        "The app crashes every time I open the settings page on Android 15.",
        "Production is DOWN. All API calls return 502 since the 14:00 deploy. Customers are furious.",
        "Please delete my account and all my data immediately, I am done with this service.",
        "My monstera's lower leaves are turning yellow and mushy, and the soil is still soggy.",
        "Roots are poking out of the drainage holes of the peace lily and water runs straight through.",
    ]
    questions = {
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                    "account": "login, profile, deletion", "other": "everything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "blocking"]},
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
        "needs_repot": {"type": "noul", "instructions": "Has the plant outgrown its pot?"},
    }

    def decision(ans):
        """The answer's top option, and its margin over the runner-up (for spotting near-ties)."""
        if ans["type"] == "noul":
            p = [1 - ans["noul"], ans["noul"]]
        else:
            p = list(ans["probabilities"].values())
        top = sorted(range(len(p)), key=p.__getitem__, reverse=True)
        return top[0], p[top[0]] - p[top[1]], p

    agree = total = 0
    worst = 0.0
    real_mismatch = False
    for state in states:
        ours, ref = decider.predict(state, questions), reference.predict(state, questions)
        for qid in questions:
            (a, margin, pa), (b, _, pb) = decision(ref["answers"][qid]), decision(ours["answers"][qid])
            total += 1
            agree += a == b
            worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
            if a != b:
                # A flip where PyTorch itself was nearly undecided is precision noise; one where it
                # was confident means the engine is wrong.
                kind = "near-tie" if margin < 0.1 else "MISMATCH"
                real_mismatch |= kind == "MISMATCH"
                print(f"  {kind}: {qid} on {state[:50]!r} (PyTorch margin {margin:.3f})")
    print(f"  {agree}/{total} decisions match stock Laya (PyTorch); largest probability difference {worst:.4f}")
    return not real_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", default="convaiinnovations/laya", help="Hub id or local checkpoint directory")
    ap.add_argument("--dtype", choices=DTYPES, default="fp16",
                    help="engine precision; fp16 was the fastest and most accurate half precision on laya")
    ap.add_argument("--output", type=Path, help="bundle directory (default: engines/<checkpoint>-<dtype>)")
    g = ap.add_argument_group("optimization profile (requests outside [min, max] are rejected)")
    g.add_argument("--opt-batch", type=int, default=16, help="rows (states x questions) to tune for")
    g.add_argument("--max-batch", type=int, default=64, help="most rows per request")
    g.add_argument("--opt-seq", type=int, default=128, help="tokens per row to tune for")
    g.add_argument("--max-seq", type=int, help="most tokens per row (default: the checkpoint's max_len)")
    g.add_argument("--opt-options", type=int, default=4, help="options per question to tune for")
    g.add_argument("--max-options", type=int, default=64, help="most options per question")
    ap.add_argument("--no-tf32", action="store_true", help="fp32 only: disable TF32 matmuls (slower, exact)")
    ap.add_argument("--workspace-gb", type=int, default=4, help="builder scratch memory")
    ap.add_argument("--keep-onnx", action="store_true", help="keep the exported ONNX in build/")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="TensorRT verbose logging")
    args = ap.parse_args()

    # Resolve the checkpoint and derive the optimization profile ranges from the CLI args and config.
    ckpt_dir = serve.resolve_checkpoint(args.checkpoint)
    revision = None if Path(args.checkpoint).is_dir() else ckpt_dir.name   # a Hub snapshot dir is named by its commit hash
    cfg = serve.Calibration.model_validate_json((ckpt_dir / "rl_agent_config.json").read_text())
    max_seq = args.max_seq or cfg.max_len
    profile = {"batch": (1, args.opt_batch, args.max_batch),
               "seq": (8, args.opt_seq, max_seq),
               "options": (2, args.opt_options, args.max_options)}

    # Reject profiles where opt falls outside [min, max].
    for dim, (lo, opt, hi) in profile.items():
        if not lo <= opt <= hi:
            raise SystemExit(f"{dim}: need {lo} <= opt ({opt}) <= max ({hi})")

    name = f"{Path(args.checkpoint.rstrip('/')).name}-{args.dtype}"
    bundle = args.output or HERE / "engines" / name
    work = HERE / "build" / name
    onnx_path = work / "model.onnx"
    bundle.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Stage 1: export the PyTorch model to ONNX at the requested precision.
    print(f"[1/4] Exporting {args.checkpoint} to ONNX ({args.dtype}) ...")
    export_onnx(args.checkpoint, args.dtype, onnx_path)

    # Stage 2: compile the ONNX graph into a TensorRT engine, then clean up the intermediate files.
    print(f"[2/4] Building the TensorRT engine for {torch.cuda.get_device_name()} ...")
    build_engine(onnx_path, bundle / "model.engine", profile, not args.no_tf32, args.workspace_gb, args.verbose)
    if not args.keep_onnx:
        shutil.rmtree(work)

    # Stage 3: assemble the bundle (tokenizer, config, and build.json metadata) next to the engine.
    print(f"[3/4] Writing bundle {bundle}")
    shutil.copytree(ckpt_dir / "tokenizer", bundle / "tokenizer", dirs_exist_ok=True)
    shutil.copy(ckpt_dir / "rl_agent_config.json", bundle / "rl_agent_config.json")
    import laya
    major, minor = torch.cuda.get_device_capability()
    (bundle / "build.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "revision": revision, "dtype": args.dtype,
        "tf32": not args.no_tf32 if args.dtype == "fp32" else None,
        "profile": {d: dict(zip(("min", "opt", "max"), v)) for d, v in profile.items()},
        "tensorrt": trt.__version__, "laya": laya.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": f"{major}.{minor}",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    size = (bundle / "model.engine").stat().st_size / 1e6
    print(f"      model.engine {size:.0f} MB, built in {time.perf_counter() - t0:.0f}s total")

    # Stage 4: free GPU memory, then check the engine against stock Laya (unless skipped).
    if args.skip_verify:
        print("[4/4] Skipped verification")
    else:
        gc.collect()
        torch.cuda.empty_cache()
        print("[4/4] Verifying against stock Laya ...")
        if not verify(bundle, args.checkpoint):
            raise SystemExit("verification failed: the engine disagrees with PyTorch on a confident decision")
    print(f"\nReady:  uv run serve.py --engine {bundle.relative_to(HERE) if bundle.is_relative_to(HERE) else bundle}")


if __name__ == "__main__":
    main()
