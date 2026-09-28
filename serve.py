"""A standalone, Jev-compatible `POST /v1/systemone` server for a Laya TensorRT engine.

This file does not import `laya`. It needs two things:

  1. A TensorRT engine built from a Laya checkpoint.
  2. That checkpoint's tokenizer and calibration config (`tokenizer/tokenizer.json` and
     `rl_agent_config.json`). The engine only contains the neural network; turning text into
     token ids, and logits into calibrated probabilities, needs these two files.

`build_engine.py` writes both into one bundle directory (model.engine, tokenizer/,
rl_agent_config.json, build.json), so the usual way to run is simply:

    uv run serve.py --engine engines/laya-fp16
    LAYA_API_KEY=secret uv run serve.py --engine engines/laya-fp16    # require a bearer token

A bare .engine file also works; the tokenizer and config then come from --checkpoint, a Hub id
(downloaded, a few MB) or a local checkpoint directory:

    uv run serve.py --engine my.engine --checkpoint convaiinnovations/laya --port 9000

Everything Laya's Python library does around the network -- validating questions, laying out
the token sequence, decoding answers -- is reimplemented below, following laya 0.3.21.

Wire protocol: the same request/response JSON, status codes and timing headers as Laya's own
`laya-serve` (TypeSafe Jev's /v1/systemone). Differences, because one English engine sits behind it:
  * Text that looks non-English is refused with a 422 (set LAYA_TRT_ALLOW_NON_ENGLISH=1 to answer
    anyway). This is a simpler check than laya-serve's language routing; see `looks_non_english`.
  * The request's `model` field is accepted and ignored.
  * Limits (questions per request, options per question, max_len) come from the engine's
    optimization profile. `GET /health` reports them.
"""
import argparse
import asyncio
import hmac
import json
import logging
import math
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, List, Optional

import numpy as np
import tensorrt as trt
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import (BaseModel, BeforeValidator, Field, StrictInt, ValidationError, field_validator,
                      model_validator)
from tokenizers import Tokenizer

log = logging.getLogger("laya-trt-serve")

# ------------------------------------------------------------------------------------------------
# 1. The TensorRT engine
# ------------------------------------------------------------------------------------------------
# The engine's five inputs and two outputs, as named when the network was exported to ONNX.
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16,
                trt.int64: torch.int64, trt.int32: torch.int32, trt.bool: torch.bool}


class Engine:
    """Runs the engine on numpy inputs, using torch CUDA tensors as the GPU buffers."""

    def __init__(self, path: Path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load the serialized engine from disk (fails if built for another GPU or TensorRT version).
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not load {path}: built for a different GPU or TensorRT version?")

        # The execution context holds per-inference state (shapes, tensor addresses).
        self.context =self.engine.create_execution_context()

        # A dedicated CUDA stream for copies and inference.
        self.stream = torch.cuda.Stream()

        # Discover the output tensors and each tensor's dtype, and check this is a Laya engine.
        names =[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.dtypes = {n: TRT_TO_TORCH[self.engine.get_tensor_dtype(n)] for n in names}
        missing = set(INPUT_NAMES) - set(names)
        if missing:
            raise SystemExit(f"{path} is not a Laya engine: missing inputs {sorted(missing)}")

    def limits(self) -> Dict[str, int]:
        """Request limits, read from optimization profile 0."""
        lo, _, hi = self.engine.get_tensor_profile_shape("input_ids", 0)
        lo_m, _, hi_m = self.engine.get_tensor_profile_shape("marker_pos", 0)
        return {"max_rows": hi[0], "max_tokens": hi[1], "min_tokens": lo[1],
                "min_options": lo_m[1], "max_options": hi_m[1]}

    def __call__(self, feed: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Queue the input copies and the inference on our stream.
        with torch.cuda.stream(self.stream):
            keep = []   # holds the input tensors alive until the stream finishes
            # Copy each input to the GPU, declare its shape, and point the engine at it.
            for name in INPUT_NAMES:
                t = torch.from_numpy(feed[name]).to("cuda", self.dtypes[name], non_blocking=True)
                keep.append(t)
                # Returns False (does not raise) for a shape outside the profile; never ignore it.
                if not self.context.set_input_shape(name, tuple(t.shape)):
                    raise ValueError(f"{name} shape {tuple(t.shape)} is outside the engine's profile")
                self.context.set_tensor_address(name, t.data_ptr())
            out = {}   # output name -> GPU tensor the engine writes into
            # Allocate each output at the size the engine now reports for these input shapes.
            for name in self.outputs:
                out[name] = torch.empty(tuple(self.context.get_tensor_shape(name)),
                                        dtype=self.dtypes[name], device="cuda")
                self.context.set_tensor_address(name, out[name].data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        # Wait for the GPU to finish before reading results.
        self.stream.synchronize()
        # Hand back fp32 numpy arrays on the CPU.
        return {n: t.float().cpu().numpy() for n, t in out.items()}


# ------------------------------------------------------------------------------------------------
# 2. Questions: validation and normalisation
# ------------------------------------------------------------------------------------------------

class Question(BaseModel):
    """One validated question; a subclass per type. Its validators reject what Laya cannot answer.

    Validator messages leave out the question id: `parse_question` adds it.
    """
    type: ClassVar[str]
    qtype: ClassVar[int]   # the integer the network takes for this type

    instructions: Any   # any JSON; a non-string becomes compact JSON

    @model_validator(mode="before")
    @classmethod
    def _check_keys(cls, data):
        if isinstance(data, dict):
            if "instructions" not in data:
                raise ValueError("no 'instructions'; add the text the model should answer")
            if "labels" in data and cls.type != "noul":
                raise ValueError("'labels' is only supported for noul questions")
        return data

    @field_validator("instructions")
    @classmethod
    def _instructions_text(cls, v):
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


class ChoiceQuestion(Question):
    type: ClassVar[str] = "choice"
    qtype: ClassVar[int] = 0
    criteria: Any = Field(None, validate_default=True)   # normalised to {label: description or None}

    @field_validator("criteria")
    @classmethod
    def _labels(cls, crit):
        if not isinstance(crit, (dict, list)) or not crit:
            raise ValueError("a choice question takes 'criteria' as a non-empty dict "
                             "of label -> description, or a list of labels")
        seen = {}
        for i, label in enumerate(crit):
            # A label is both option text and the answer key, so it must be a hashable scalar.
            if label is None or isinstance(label, (list, dict, set, bytearray)):
                raise ValueError(f"choice label {i} must be a string, number or bool, got {label!r}")
            if label in seen:
                raise ValueError(f"choice label {i} ({label!r}) repeats label {seen[label]}; "
                                 "every option needs its own label")
            seen[label] = i
        return crit if isinstance(crit, dict) else {c: None for c in crit}

    def option_texts(self) -> List[str]:
        return [str(k) if v is None or v == "" else f"{k}: {as_text(v)}" for k, v in self.criteria.items()]


class ScoreQuestion(Question):
    type: ClassVar[str] = "score"
    qtype: ClassVar[int] = 1
    criteria: Any = Field(None, validate_default=True)   # level descriptions, index 0 first

    @field_validator("criteria")
    @classmethod
    def _levels(cls, crit):
        if not isinstance(crit, list) or not crit:
            raise ValueError("a score question takes 'criteria' as a non-empty list "
                             "of level descriptions, index 0 first")
        if None in crit:
            raise ValueError(f"score level {crit.index(None)} is null")
        return crit

    def option_texts(self) -> List[str]:
        return [f"level {i}: {as_text(c)}" for i, c in enumerate(self.criteria)]


class NoulQuestion(Question):
    type: ClassVar[str] = "noul"
    qtype: ClassVar[int] = 2
    criteria: Any = None   # optional, keyed 'true'/'false'; keys are lower-cased
    labels: Any = Field(None, validate_default=True)   # normalised to a (false label, true label) pair

    @field_validator("criteria")
    @classmethod
    def _keys(cls, crit):
        if crit is None:
            return None
        if not isinstance(crit, dict) or not {str(k).lower() for k in crit} <= {"true", "false"}:
            raise ValueError("a noul question takes 'criteria' keyed only 'true'/'false', or omits it")
        return {str(k).lower(): v for k, v in crit.items()}

    @field_validator("labels")
    @classmethod
    def _label_pair(cls, labels):
        """The two option labels of a yes/no question; 'false'/'true' unless the caller renamed them."""
        if labels is None:
            return "false", "true"
        bad = ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
        if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
            raise bad
        f, t = labels["false"], labels["true"]
        if not isinstance(f, str) or not isinstance(t, str) or not f.strip() or not t.strip() \
                or f.strip() == t.strip():
            raise bad
        return f.strip(), t.strip()

    def option_texts(self) -> List[str]:
        crit = self.criteria or {}
        f_label, t_label = self.labels
        f_crit, t_crit = crit.get("false"), crit.get("true")
        return [f"{f_label}: " + (as_text(f_crit) if f_crit not in (None, "") else "no, the statement does not hold"),
                f"{t_label}: " + (as_text(t_crit) if t_crit not in (None, "") else "yes, the statement holds")]


QUESTION_TYPES = {c.type: c for c in (ChoiceQuestion, ScoreQuestion, NoulQuestion)}


def parse_question(qid: str, raw: Any) -> Question:
    """Validate one question definition into its model, naming the question on failure (a 422)."""
    if not isinstance(raw, dict):
        raise ValueError(f"question {qid!r}: definition must be a dict, got {type(raw).__name__}")
    t = raw.get("type")
    cls = QUESTION_TYPES.get(t) if isinstance(t, str) else None
    if cls is None:
        raise ValueError(f"question {qid!r}: unknown type {t!r}; use one of {sorted(QUESTION_TYPES)}")
    try:
        return cls.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"question {qid!r}: {e.errors()[0]['msg'].removeprefix('Value error, ')}") from e


def as_text(value) -> str:
    """Strings pass through; anything structured becomes compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def state_text(state) -> str:
    """The state as text: a string as is, a conversation as JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


# ------------------------------------------------------------------------------------------------
# 3. Tokenization: (state, question) -> one row of token ids
# ------------------------------------------------------------------------------------------------
class TextEncoder:
    def __init__(self, checkpoint_dir: Path):
        # Load the checkpoint's tokenizer.
        self.tok = Tokenizer.from_file(str(checkpoint_dir / "tokenizer" / "tokenizer.json"))
        # tokenizer.json can carry a saved truncation/padding setting; we manage lengths ourselves.
        self.tok.no_truncation()
        self.tok.no_padding()
        special = json.loads((checkpoint_dir / "tokenizer" / "tokenizer_config.json").read_text())
        self.mask_text = special["mask_token"]
        self.cls, self.sep, self.mask, self.pad = (self.tok.token_to_id(special[k]) for k in
                                                   ("cls_token", "sep_token", "mask_token", "pad_token"))

    def ids(self, text: str) -> List[int]:
        # A literal "[MASK]" in user text would be read as an option marker, so blank it out.
        return self.tok.encode(text.replace(self.mask_text, " "), add_special_tokens=False).ids

    def row(self, state_ids: List[int], q: Question, max_len: int, head_max_len: int, truncate_left: bool):
        """Lay out one row:  [CLS] <type> question: ... [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

        The question and its options come first and share a budget of `head_max_len` tokens; the
        state fills whatever is left of `max_len`. Returns (ids, marker positions, option stats).
        """
        # Tokenize the question head and each option (with its [MASK] marker), and see what budget is left.
        opts = q.option_texts()
        head = self.ids(f"{q.type} question: {q.instructions}")
        opt_ids = [[self.mask] + self.ids(" " + o)[:48] for o in opts]   # each option capped at 48
        budget = head_max_len - sum(len(o) for o in opt_ids)
        per_option = None
        # If the options leave under 16 tokens for the question text, shrink them.
        if budget < 16:
            # Too many / too long options: cut every option to an equal share of the head budget.
            per_option = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per_option] for o in opt_ids]
            budget = head_max_len - sum(len(o) for o in opt_ids)

        # Assemble the head, recording where each option's [MASK] marker lands, then append the state.
        ids = [self.cls] + head[:max(8, budget)] + [self.sep]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep)
        room = max(0, max_len - len(ids) - 1)
        # Conversations (lists) keep their newest turns, so they are cut from the left.
        st = state_ids[max(0, len(state_ids) - room):] if truncate_left else state_ids[:room]
        ids = (ids + st + [self.sep])[:max_len]
        markers = [m for m in markers if m < max_len]
        if len(markers) != len(opts):
            raise ValueError(f"the options of a {q.type} question do not fit in head_max_len={head_max_len}")
        stats = {"options": len(opt_ids), "options_distinct": len({tuple(o) for o in opt_ids}),
                 "tokens_per_option": per_option}
        # The stats let the caller report options that were truncated into indistinguishability.
        return ids, markers, stats


# ------------------------------------------------------------------------------------------------
# 4. Decoding: logits -> calibrated, typed answers
# ------------------------------------------------------------------------------------------------
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def clamp_temperature(t) -> float:
    """Keep a calibration temperature in [0.5, 5]; a bad value falls back to 1."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


Temperature = Annotated[float, BeforeValidator(clamp_temperature)]


class Calibration(BaseModel):
    """rl_agent_config.json: sequence limits and calibration temperatures (other keys are ignored)."""
    max_len: int = 512
    head_max_len: int = 192
    temperature: List[Temperature] = [1.0] * 3   # by question type, indexed by Question.qtype
    temperature_by_options: Dict[str, Temperature] = {}   # by "<type>:<option-count bucket>"


def temperature_bucket(t: str, k: int) -> str:
    return f"{t}:{'2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'}"


def decode(q: Question, logits: np.ndarray, act_p: float, cfg: Calibration) -> Dict[str, Any]:
    """Turn one row's option logits into the answer dict Jev clients expect."""
    k = len(logits)
    # Temperature scaling: divide logits by a per-question-kind temperature fitted at training
    # time, so that the probabilities are calibrated.
    t_scale = cfg.temperature_by_options.get(temperature_bucket(q.type, k), cfg.temperature[q.qtype])
    z = logits / t_scale
    p = np.exp(z - z.max())
    p = p / p.sum()
    answer_conf = round(float(np.clip(p.max(), 0.0, 1.0)), 4)
    # "confidence" for choice/score is 1 - normalised entropy: how concentrated p is.
    entropy_conf = 1.0 if k < 2 else float(np.clip(
        1.0 - (-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()) / math.log(k), 0.0, 1.0))
    common = {"answer_confidence": answer_conf, "action": {"act_probability": round(act_p, 4)}}
    # Choice: the most probable label, plus per-label probabilities.
    if q.type == "choice":
        keys = list(q.criteria)
        return {"type": "choice", "choice": keys[int(p.argmax())],
                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                "confidence": round(entropy_conf, 4), **common}
    # Score: the expected level under p, plus per-level probabilities. (Anything else is noul, below.)
    if q.type == "score":
        return {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q.criteria)},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": round(entropy_conf, 4), **common}
    p_true = float(p[1])
    return {"type": "noul", "noul": round(p_true, 4), "confidence": round(max(p_true, 1 - p_true), 4),
            **common}


# ------------------------------------------------------------------------------------------------
# 5. The whole pipeline: request -> rows -> engine -> answers
# ------------------------------------------------------------------------------------------------
class Decider:
    def __init__(self, engine_path: Path, checkpoint_dir: Path):
        self.engine = Engine(engine_path)
        self.text = TextEncoder(checkpoint_dir)
        self.cfg = Calibration.model_validate_json((checkpoint_dir / "rl_agent_config.json").read_text())

    def predict(self, state, questions: Dict[str, Any], max_len=None, head_max_len=None) -> Dict[str, Any]:
        ids = list(questions)
        # No questions, nothing to run.
        if not ids:
            return {"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        # Validate every question up front so a bad one fails the request before any inference.
        qs = [parse_question(qid, questions[qid]) for qid in ids]
        max_len = self.cfg.max_len if max_len is None else max_len
        head_max_len = self.cfg.head_max_len if head_max_len is None else head_max_len

        # The state is tokenized once and shared by every question's row.
        state_ids = self.text.ids(state_text(state))
        rows = [self.text.row(state_ids, q, max_len, head_max_len, truncate_left=isinstance(state, list))
                for q in qs]

        # Collate: pad every row to the longest one, and every marker list to the most options.
        n, L, K = len(rows), max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
        feed = {"input_ids": np.full((n, L), self.text.pad, np.int64),
                "attention_mask": np.zeros((n, L), np.int64),
                "marker_pos": np.zeros((n, K), np.int64),
                "marker_mask": np.zeros((n, K), bool),
                "qtype": np.array([q.qtype for q in qs], np.int64)}
        for i, (tok_ids, markers, _) in enumerate(rows):
            feed["input_ids"][i, :len(tok_ids)] = tok_ids
            feed["attention_mask"][i, :len(tok_ids)] = 1
            feed["marker_pos"][i, :len(markers)] = markers
            feed["marker_mask"][i, :len(markers)] = True

        # Run the engine, and turn the action logits into a per-row "act" probability.
        out = self.engine(feed)
        act = out["act_logits"]
        act_p = np.exp(act - act.max(-1, keepdims=True))
        act_p = act_p / act_p.sum(-1, keepdims=True)

        # Decode each row's logits (dropping padding options) into its typed answer.
        answers = {qid: decode(q, out["logits"][i, :len(rows[i][1])], float(act_p[i, 0]), self.cfg)
                   for i, (qid, q) in enumerate(zip(ids, qs))}
        usage = {"input_tokens": int(feed["attention_mask"].sum()), "output_tokens": 0}
        # Report questions whose options got cut until some became indistinguishable.
        collapsed = {qid: {"total": s["options"], "distinct": s["options_distinct"],
                           "tokens_per_option": s["tokens_per_option"]}
                     for qid, (_, _, s) in zip(ids, rows) if s["options_distinct"] < s["options"]}
        if collapsed:
            usage["options"] = collapsed
        return {"model": "laya-rl-agent", "answers": answers, "usage": usage}


def looks_non_english(state) -> Optional[str]:
    """A guard for an English-only engine. Returns a reason to refuse, or None.
    """
    text = state_text(state)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    non_ascii = [c for c in letters if ord(c) > 127]
    rate = len(non_ascii) / len(letters)
    # Under 2% non-ASCII letters is tolerated (names, accents, the odd symbol).
    if rate < 0.02:
        return None
    scripts = {unicodedata.name(c, "UNKNOWN").split()[0] for c in non_ascii}
    return f"{rate:.0%} of letters are non-English ({', '.join(sorted(scripts))})"


# ------------------------------------------------------------------------------------------------
# 6. The HTTP server
# ------------------------------------------------------------------------------------------------
# Request size caps: raw body bytes, and the state's length in characters.
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_STATE_CHARS = 50_000


class SystemOneRequest(BaseModel):
    """The POST /v1/systemone body (other keys, like `model`, are ignored).

    Questions stay raw here; `Decider.predict` validates them once the size limits have passed.
    """
    questions: Dict[str, Any]
    state: Any   # a string, or a conversation (list of turns)
    max_len: Optional[StrictInt] = Field(None, gt=0)
    head_max_len: Optional[StrictInt] = Field(None, gt=0)

    @field_validator("state")
    @classmethod
    def _state_given(cls, v):
        if v is None:
            raise ValueError("'state' is required")
        return v


def parse_request(body: Any) -> SystemOneRequest:
    """Validate the body's structure, mapping the first failure to the status Jev clients expect."""
    try:
        return SystemOneRequest.model_validate(body)
    except ValidationError as e:
        err = e.errors()[0]
        field = err["loc"][0] if err["loc"] else None
        if field in ("max_len", "head_max_len"):
            raise HTTPException(422, f"{field} must be a positive integer")
        if field == "state":
            raise HTTPException(400, "'state' is required")
        if field == "questions" and err["type"] != "missing":
            raise HTTPException(400, "'questions' must be an object")
        raise HTTPException(400, "request body must be an object with a 'questions' field")


def check_request(body: Any, limits: Dict[str, int]) -> SystemOneRequest:
    """Cheap checks before any tokenization (400/413/422). Returns the parsed request."""
    req = parse_request(body)
    questions = req.questions
    size = len(state_text(req.state))
    if size > MAX_STATE_CHARS:
        raise HTTPException(413, f"state too large ({size} > {MAX_STATE_CHARS} chars)")
    # Each question is one row of the engine's batch; each option is one marker.
    if len(questions) > limits["max_rows"]:
        raise HTTPException(413, f"too many questions ({len(questions)} > {limits['max_rows']}, "
                                 "the engine's max batch)")
    for qid, q in questions.items():
        if isinstance(q, dict) and q.get("type") in ("choice", "score") and isinstance(q.get("criteria"), (dict, list)):
            k = len(q["criteria"])
            if k > limits["max_options"]:
                raise HTTPException(413, f"question {qid!r} has {k} options; the engine accepts at most "
                                         f"{limits['max_options']}")
            if 0 < k < limits["min_options"]:
                raise HTTPException(422, f"question {qid!r} has {k} option(s); the engine needs at least "
                                         f"{limits['min_options']}")
    for key in ("max_len", "head_max_len"):
        val = getattr(req, key)
        if val is not None and val > limits["max_tokens"]:
            raise HTTPException(422, f"{key} exceeds the engine's max sequence length "
                                     f"({val} > {limits['max_tokens']})")
    return req


def create_app(engine_path: Path, checkpoint_dir: Path) -> FastAPI:
    # Load the model and gather the server's settings (limits, build info, auth, concurrency).
    decider = Decider(engine_path, checkpoint_dir)
    limits = decider.engine.limits()
    # A bundle is named by its directory (engines/laya-fp16/model.engine -> "laya-fp16").
    name = engine_path.parent.name if engine_path.name == "model.engine" else engine_path.stem
    build_info_path = engine_path.parent / "build.json"
    build_info = json.loads(build_info_path.read_text()) if build_info_path.exists() else None
    allow_non_english = os.environ.get("LAYA_TRT_ALLOW_NON_ENGLISH", "") in ("1", "true", "yes")
    api_key = os.environ.get("LAYA_API_KEY") or None
    # Compare bytes: hmac.compare_digest raises on a non-ASCII str, which would be a 500, not a 401.
    expected_auth = f"Bearer {api_key}".encode("utf-8", "surrogateescape") if api_key else b""
    # One execution context, one GPU: inference runs on a single worker thread, one at a time, off
    # the event loop so /health stays responsive.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt-infer")
    max_concurrent = int(os.environ.get("LAYA_MAX_CONCURRENT", 16))
    admission = asyncio.Semaphore(max_concurrent)

    # Warm up so the first real request doesn't pay for lazy CUDA initialisation.
    decider.predict("warm up", {"q": {"type": "noul", "instructions": "Is this a test?"}})

    @asynccontextmanager
    async def lifespan(_app):
        yield
        pool.shutdown(wait=True, cancel_futures=True)

    # The FastAPI app; its lifespan shuts the inference pool down on exit.
    app = FastAPI(title="laya-trt-serve", lifespan=lifespan,
                  summary="Laya decisions over the Jev /v1/systemone protocol, on TensorRT")

    # Liveness check that also reports the engine, GPU, request limits and build info.
    @app.get("/health")
    def health():
        return {"status": "ok", "backend": "tensorrt", "engine": name,
                "device": torch.cuda.get_device_name(), "limits": limits, "build": build_info}

    # The main endpoint: authenticate, apply the concurrency cap, then hand off to `handle`.
    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        if api_key is not None:
            supplied = (authorization or "").encode("utf-8", "surrogateescape")
            if not hmac.compare_digest(supplied, expected_auth):
                raise HTTPException(401, "invalid or missing bearer token")
        # Past the cap, refuse instead of queueing, so buffered request bodies stay bounded.
        if admission.locked():
            raise HTTPException(503, "server busy, try again later", headers={"Retry-After": "1"})
        async with admission:
            return await handle(request)

    # Read and validate the request, run inference on the worker thread, and build the timed response.
    async def handle(request: Request):
        chunks, total = [], 0
        async for chunk in request.stream():   # enforce the cap whatever Content-Length says
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                raise HTTPException(413, "request body too large")
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except (ValueError, RecursionError):
            raise HTTPException(400, "request body must be valid JSON")
        req = check_request(body, limits)
        overrides = req.model_dump(include={"max_len", "head_max_len"}, exclude_none=True)
        if not allow_non_english:
            reason = looks_non_english(req.state)
            if reason:
                raise HTTPException(422, f"this server runs an English-only engine: {reason}")

        t0 = time.perf_counter()
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                pool, lambda: decider.predict(req.state, req.questions, **overrides))
        except ValueError as e:
            # Question validation or an engine-profile violation: messages are safe to return.
            raise HTTPException(422, str(e))
        except Exception:
            log.exception("inference failed")   # the details go to the log, never to the client
            raise HTTPException(500, "inference failed")
        ms = (time.perf_counter() - t0) * 1000
        # log.warning(f"Prediction Handle Time: {ms:0.3f}ms")
        return JSONResponse(result, headers={"Server-Timing": f"inference;dur={ms:.2f}",
                                             "X-Inference-Time-Ms": f"{ms:.2f}",
                                             "X-Backend": f"tensorrt:{name}"})

    return app


def resolve_checkpoint(name: str) -> Path:
    """A local checkpoint directory, or a Hub id whose tokenizer and config we download."""
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download
    # Fetch only the small files serve.py needs, not the model weights.
    return Path(snapshot_download(name, allow_patterns=["rl_agent_config.json", "tokenizer/*"]))


def resolve_engine(engine: Path, checkpoint: Optional[str]):
    """Return (engine file, checkpoint dir) for a bundle directory or a bare .engine file."""
    engine_file = engine / "model.engine" if engine.is_dir() else engine
    if not engine_file.is_file():
        raise SystemExit(f"no engine at {engine_file}")
    if checkpoint:
        return engine_file, resolve_checkpoint(checkpoint)
    bundle = engine_file.parent
    if (bundle / "rl_agent_config.json").is_file() and (bundle / "tokenizer" / "tokenizer.json").is_file():
        return engine_file, bundle
    raise SystemExit(f"{engine_file} has no tokenizer/ and rl_agent_config.json beside it; pass "
                     "--checkpoint with the Hub id or directory of the checkpoint it was built from")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jev-compatible /v1/systemone server for a Laya TensorRT engine")
    parser.add_argument("--engine", required=True, type=Path,
                        help="a bundle directory from build_engine.py, or a bare .engine file")
    parser.add_argument("--checkpoint", default=None,
                        help="only for a bare .engine: the checkpoint it was built from, as a Hub id or a "
                             "local directory (only its tokenizer and rl_agent_config.json are read)")
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", 8000)))
    args = parser.parse_args()
    uvicorn.run(create_app(*resolve_engine(args.engine, args.checkpoint)), host=args.host, port=args.port)
