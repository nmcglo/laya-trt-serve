# laya-trt-serve

This project was intended to help refamiliarize myself with TensorRT and apply it to the new hotness (which is just a revisiting of classical language model inference...) of system-one models like Jev - using the open model Laya.
This project was developed with assistance of my local AI stack running Qwen3.8:27b with the Oh-My-Pi harness.

Serve the [Laya](https://huggingface.co/convaiinnovations/laya) decision model on NVIDIA TensorRT, behind a Jev-compatible `POST /v1/systemone` API (the same wire protocol as Laya's own `laya-serve`).

There are two scripts:

- **`build_engine.py`** turns a Laya checkpoint into an engine **bundle**: a directory holding the TensorRT engine plus the checkpoint's tokenizer and calibration config. It uses `laya`, `onnx` and `onnxscript`, which live in the `build` dependency group.
- **`serve.py`** serves a bundle. It doesn't import `laya`: it needs only the bundle and the runtime dependencies (`tensorrt`, `torch`, `tokenizers`, `fastapi`, …).

## Requirements

- Linux with an NVIDIA GPU and driver; the `tensorrt` and `torch` wheels bring their own CUDA libraries.
- [uv](https://docs.astral.sh/uv/).

## Build an engine

```bash
uv run --group build build_engine.py
```

This takes about a minute and writes `engines/laya-fp16/`:

```
engines/laya-fp16/
  model.engine            # the compiled network (848 MB for fp16)
  tokenizer/              # from the checkpoint: text -> token ids
  rl_agent_config.json    # from the checkpoint: calibration temperatures, max_len
  build.json              # how it was built: checkpoint + revision, dtype, profile, versions, GPU
```

The build exports the checkpoint to ONNX, compiles it with TensorRT, writes the bundle, and then **verifies** it: sample questions answered through `serve.py`'s pipeline must reach the same decisions as stock Laya in PyTorch. On an RTX 3080 Ti, 24/24 decisions matched, with a largest probability difference of 0.012.

Useful options (see `--help`):

| option | default | meaning |
|---|---|---|
| `--dtype` | `fp16` | `fp16`, `bf16` or `fp32`. On laya, fp16 was the fastest and the most accurate half precision |
| `--checkpoint` | `convaiinnovations/laya` | Hub id or local checkpoint directory |
| `--max-batch` / `--opt-batch` | 64 / 16 | rows per request (a row is one state × one question) |
| `--max-seq` / `--opt-seq` | checkpoint `max_len` (512) / 128 | tokens per row |
| `--max-options` / `--opt-options` | 64 / 4 | options per question (the minimum is fixed at 2) |
| `--output` | `engines/<checkpoint>-<dtype>` | bundle directory |

Requests outside the `min`–`max` ranges are rejected. `opt` is the size TensorRT tunes its kernels for, so set it to your most common request.

**Build on the machine that will serve.** An engine only runs on the GPU architecture and TensorRT version that built it.

## Serve

```bash
uv run serve.py --engine engines/laya-fp16
```

The server listens on `127.0.0.1:8000`; pass `--host 0.0.0.0` to expose it.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, I want my money back",
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "billing and refunds", "tech": "login and app issues", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["calm", "firm", "angry", "furious"]},
    "churn":   {"type": "noul", "instructions": "Does the customer threaten to leave?"}
  }
}'
```

`GET /health` reports the GPU, the request limits read from the engine, and `build.json`.

| env var | meaning | default |
|---|---|---|
| `LAYA_API_KEY` | if set, require `Authorization: Bearer <key>` on `/v1/systemone` | none |
| `LAYA_MAX_CONCURRENT` | requests admitted at once; the rest get `503` | 16 |
| `LAYA_TRT_ALLOW_NON_ENGLISH` | `1` to answer text that looks non-English | off |
| `LAYA_HOST` / `LAYA_PORT` | bind address, as defaults for `--host` / `--port` | `127.0.0.1` / `8000` |

A bare `.engine` file also works with `--checkpoint <hub id or dir>`, which supplies the tokenizer and config. They must come from the checkpoint the engine was built from.

## How it differs from `laya-serve`

- **One engine, English only.** `laya-serve` routes non-English text to `laya-multilingual`. This server refuses text where at least 2% of the letters are non-ASCII, with a 422. That's `laya-serve`'s threshold without its language identification, so a short English text with one accented word ("Café closed") can be refused too.
- **The `model` field** in a request is accepted and ignored.
- **No `routing` block** in the response. `answers` and `usage`, the parts Jev clients read, are the same; the engine is named in the `X-Backend` header.
- **Limits come from the engine** (questions ≤ max batch, options within the profile, `max_len` ≤ max sequence length) rather than fixed constants.
- **Pre- and post-processing follow laya 0.3.21.** `serve.py` reimplements Laya's question validation, token layout and answer decoding. Against laya 0.3.21 on the same engine it produced byte-identical responses across 16 edge cases. A future Laya release that changes these would need the same changes here.

## Performance

About 8 ms per request (1 state, 3 questions) on an RTX 3080 Ti, against about 13.5 ms for stock Laya in PyTorch on the same GPU. Requests run one at a time on a single inference thread, so throughput is roughly 1000 / latency ≈ 120–140 requests/s. Batching concurrent requests together would raise that considerably, because at this size the GPU is mostly idle.
