"""YAML config -> dataclasses for a comparison run.

A run has one `reference` and a **list** of `candidates`: the reference is
scored once and every candidate is compared against it. The `cfg.candidate` /
`cfg.runtime.cand` read-properties return the first candidate so single-candidate
call sites (null mode, tests) stay simple.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


class Provider(str, enum.Enum):
    """Serving stacks the probe can drive. TGL (Together's internal sglang
    fork, e.g. the smg:*-tgl-* / tgl:* ECR images) speaks the sglang API and
    shares SGLangClient, but is kept a distinct provider so a mixed sglang/tgl
    pair is flagged as two different builds."""

    SGLANG = "sglang"
    TGL = "tgl"
    VLLM = "vllm"


def detect_engine(image: str) -> Provider:
    """Infer the serving provider from the docker image name: it must contain
    exactly one of the Provider tokens."""
    name = image.lower()
    matches = [p for p in Provider if p.value in name]
    if len(matches) != 1:
        found = ", ".join(p.value for p in matches) or "none"
        raise ValueError(
            f"cannot detect engine from image {image!r}: the name must contain "
            f"exactly one of {', '.join(p.value for p in Provider)} (found: {found})"
        )
    return matches[0]


def parse_gpus(value) -> list[str]:
    # bool first: YAML `gpus: yes` parses as True, and bool is a subclass of int
    if isinstance(value, bool) or (
        isinstance(value, (list, tuple)) and any(isinstance(g, bool) for g in value)
    ):
        raise ValueError(f"gpus must be a list or comma-separated string, got {value!r}")
    if isinstance(value, str):
        gpus = [g.strip() for g in value.split(",") if g.strip()]
    elif isinstance(value, int):
        gpus = [str(value)]
    elif isinstance(value, (list, tuple)):
        gpus = [str(g).strip() for g in value if str(g).strip()]
    else:
        raise ValueError(f"gpus must be a list or comma-separated string, got {value!r}")
    if not gpus:
        raise ValueError("gpus must contain at least one device id")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPU device ids: {gpus}")
    return gpus


# Ingest source formats: how a row of precomputed reference generations maps to
# (system, question, reasoning, response). The `hle_response_cache` handler
# hardcodes the original-HLE system prompts and selects by answer_type.
INGEST_FORMATS = ("hle_response_cache", "chatml_messages", "pg19")


@dataclass
class ModelSpec:
    model_path: str
    image: str = "lmsysorg/sglang:latest"
    tp: int = 1
    extra_args: str = ""
    # Candidate identity — used for artifact filenames (scores_<name>.parquet)
    # and metrics keys. The reference always uses "ref"; candidate names must be
    # unique across the candidates list.
    name: str = "cand"
    # Extra environment variables for the server process — for engine knobs
    # that have no CLI flag (e.g. SGLANG_MINIMAX_SPARSE_DECODE_BACKEND). They
    # select kernel paths, so they change what the server computes, like
    # extra_args.
    env: dict = field(default_factory=dict)
    # Loader-patch scripts (host paths to .py files) run in-container with
    # python3 before the server starts, in order — e.g. the excluded-MoE MXFP8
    # rebuild required by the MiniMax-M3 skip10-family checkpoints. A patch
    # changes the served weights.
    patches: list = field(default_factory=list)

    @property
    def engine(self) -> Provider:
        return detect_engine(self.image)


@dataclass
class ServerRuntime:
    gpus: list[str] = field(default_factory=lambda: ["0"])
    port: int = 38090
    # If set, this server is assumed to be already running and docker
    # orchestration is skipped for this side.
    base_url: str | None = None

    @property
    def url(self) -> str:
        return (self.base_url or f"http://127.0.0.1:{self.port}").rstrip("/")

    @property
    def managed(self) -> bool:
        """True if we are responsible for launching this server via docker."""
        return self.base_url is None


@dataclass
class RuntimeCfg:
    ref: ServerRuntime
    # Placement per candidate, keyed by candidate name. On a box that can't fit
    # the servers together, candidates share a GPU set and run sequentially
    # (one at a time through the single reusable candidate server slot).
    candidates: dict = field(default_factory=dict)

    @property
    def cand(self) -> ServerRuntime:
        """First candidate runtime — convenience for single-candidate call sites."""
        return next(iter(self.candidates.values()))


@dataclass
class PromptsCfg:
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    split: str = "test_sft"
    n: int = 512
    seed: int = 0
    max_prompt_tokens: int = 2048
    # Column holding the prompt text. None = auto-detect among common names
    # ("prompt", "question", "instruction", "text") or a chat "messages" column.
    column: str | None = None
    # Extra kwargs for tokenizer.apply_chat_template (e.g. enable_thinking).
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass
class IngestCfg:
    """Precomputed reference generations (allow_generations: false). The source
    provides (system, question, reasoning, response) per row; the M3 chat
    template reassembles the exact token sequence the model processed."""

    source: str = ""
    format: str = "hle_response_cache"
    # pg19 (raw-text corpus): HF split + the field holding the document text.
    split: str = "test"
    text_field: str = "text"
    # Field names inside each source JSON (hle_response_cache).
    question_field: str = "question"
    reasoning_field: str = "reasoning"
    response_field: str = "response"
    answer_type_field: str = "answer_type"
    finish_reason_field: str = "finish_reason"
    # Drop rows whose finish_reason != this (the 25 cap-truncated HLE traces
    # whose content collapsed into reasoning_content). None = keep all.
    require_finish_reason: str | None = "stop"
    # TEMPORARY smoke-test caps — removed once the streaming path is proven
    # (rationale in docs/precomputed-generations.md, "Truncation caps").
    max_prompts: int | None = None  # cap number of rows ingested
    max_score_tokens: int | None = None  # truncate each sequence's scored length


@dataclass
class GenerationCfg:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    seed: int = 0  # per-request sampling_seed base (seed + pid)


@dataclass
class ScoringCfg:
    # k of the retrieved top-k. "auto" derives from the reference's
    # generation_config.json top_k, falling back to 40 (M3 model-card top_k)
    # when absent/disabled. An int overrides. Resolved to an int at run start.
    top_logprobs_num: int | str = "auto"
    concurrency: int = 32
    # 0 = one-shot scoring (default). >0 would enable chunked scoring — deferred;
    # not implemented until a scale test shows one-shot responses break. Design:
    # docs/precomputed-generations.md, "Logprob response payload".
    score_chunk_size: int = 0


@dataclass
class MetricsCfg:
    # Per-bucket metrics: a bucket is bucket_size contiguous token positions.
    # Reporting-only — re-bucketing does not invalidate scores.
    bucket_size: int = 4096
    # Worst-KL drill-down size kept in metrics.json.
    worst_n: int = 20


@dataclass
class RunCfg:
    run_name: str
    reference: ModelSpec
    candidates: list  # list[ModelSpec]
    runtime: RuntimeCfg
    hf_home: str = "/data/huggingface"
    output_dir: str = "runs"
    # Per-invocation timestamp subdir: artifacts always land in
    # runs/<run_name>/<run_stamp>, so each run of the same config is preserved
    # separately. Generated at construction; sortable and filesystem-safe.
    run_stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # false -> ingest precomputed reference generations (no generation); true ->
    # sample prompts + generate on the reference server.
    allow_generations: bool = True
    ingest: IngestCfg | None = None
    prompts: PromptsCfg = field(default_factory=PromptsCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    @property
    def candidate(self) -> ModelSpec:
        """First candidate — convenience for single-candidate call sites."""
        return self.candidates[0]

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / self.run_stamp


def _build(cls, data: dict):
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _build_model_spec(data: dict, default_name: str = "cand") -> ModelSpec:
    raw = dict(data) if isinstance(data, dict) else data
    if not isinstance(raw, dict):
        raise ValueError("model spec must be a mapping")
    raw.setdefault("name", default_name)
    if "env" in raw:
        if not isinstance(raw["env"], dict):
            raise ValueError("env must be a mapping of VAR: value")
        env = {}
        for key, val in raw["env"].items():
            if not isinstance(key, str) or not _ENV_KEY.match(key):
                raise ValueError(f"invalid env var name: {key!r}")
            if isinstance(val, bool) or not isinstance(val, (str, int, float)):
                raise ValueError(f"env {key} must be a string or number, got {val!r}")
            env[key] = str(val)
        raw["env"] = env
    if "patches" in raw:
        if not isinstance(raw["patches"], list) or not all(
            isinstance(p, str) for p in raw["patches"]
        ):
            raise ValueError("patches must be a list of file paths")
        for p in raw["patches"]:
            if not p.endswith(".py"):
                raise ValueError(f"patches must be .py files, got {p!r}")
    return _build(ModelSpec, raw)


def _build_candidates(raw: dict) -> list:
    """Build the candidate list from `candidates:` (a non-empty list)."""
    if "candidates" not in raw:
        raise ValueError("config missing required key: candidates")
    items = raw["candidates"]
    if not isinstance(items, list) or not items:
        raise ValueError("candidates must be a non-empty list")
    specs = []
    for i, item in enumerate(items):
        # a 1-element list may omit the name (defaults to "cand"); a ladder
        # requires an explicit unique name per candidate
        default = "cand" if len(items) == 1 else None
        if default is None and not (isinstance(item, dict) and item.get("name")):
            raise ValueError(f"candidates[{i}] must have a unique 'name'")
        specs.append(_build_model_spec(item, default_name=default or "cand"))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be unique: {names}")
    return specs


def _build_runtime_side(data: dict) -> ServerRuntime:
    raw = dict(data)
    if "gpus" in raw:
        raw["gpus"] = parse_gpus(raw["gpus"])
    return _build(ServerRuntime, raw)


def _build_runtime(data: dict, candidate_names: list) -> RuntimeCfg:
    if not isinstance(data, dict):
        raise ValueError("runtime must be a mapping")
    known = {"ref", "candidates"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) for RuntimeCfg: {sorted(unknown)}")
    if "ref" not in data:
        raise ValueError("runtime missing required key: ref")
    if "candidates" not in data:
        raise ValueError("runtime missing required key: candidates")
    if not isinstance(data["candidates"], dict):
        raise ValueError("runtime.candidates must be a mapping of name -> runtime")
    candidates = {name: _build_runtime_side(rt) for name, rt in data["candidates"].items()}
    missing = set(candidate_names) - set(candidates)
    if missing:
        raise ValueError(f"runtime.candidates missing entries for: {sorted(missing)}")
    extra = set(candidates) - set(candidate_names)
    if extra:
        raise ValueError(f"runtime.candidates has unknown candidate names: {sorted(extra)}")
    return RuntimeCfg(ref=_build_runtime_side(data["ref"]), candidates=candidates)


def get_top_k(model_path: str, fallback: int = 40) -> int:
    """Read `top_k` from the model's generation_config.json (local dir first,
    then hub); return `fallback` (40) if the file is absent, unreadable, or has
    no positive top_k."""
    try:
        p = Path(model_path) / "generation_config.json"
        if p.is_file():
            data = json.loads(p.read_text())
        else:
            from huggingface_hub import hf_hub_download

            data = json.loads(
                Path(hf_hub_download(model_path, "generation_config.json")).read_text()
            )
        top_k = data.get("top_k") if isinstance(data, dict) else None
        if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
            return top_k
    except Exception:
        pass
    return fallback


def load_config(path: str | Path) -> RunCfg:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    for key in ("run_name", "reference", "runtime"):
        if key not in raw:
            raise ValueError(f"config missing required key: {key}")
    raw["reference"] = _build_model_spec(raw["reference"], default_name="ref")
    raw["candidates"] = _build_candidates(raw)
    candidate_names = [c.name for c in raw["candidates"]]
    raw["runtime"] = _build_runtime(raw["runtime"], candidate_names)
    if "ingest" in raw:
        raw["ingest"] = _build(IngestCfg, raw["ingest"])
    for key, cls in (
        ("prompts", PromptsCfg),
        ("generation", GenerationCfg),
        ("scoring", ScoringCfg),
        ("metrics", MetricsCfg),
    ):
        if key in raw:
            raw[key] = _build(cls, raw[key])
    cfg = _build(RunCfg, raw)

    # Dataclasses don't check field types; validate the scalars that would
    # otherwise fail later with confusing docker/http errors.
    specs = [("reference", cfg.reference)] + [(f"candidate {c.name}", c) for c in cfg.candidates]
    for side, spec in specs:
        if isinstance(spec.tp, bool) or not isinstance(spec.tp, int) or spec.tp < 1:
            raise ValueError(f"{side}.tp must be a positive integer, got {spec.tp!r}")
    rt_sides = [("ref", cfg.runtime.ref)] + [
        (f"candidate {n}", rt) for n, rt in cfg.runtime.candidates.items()
    ]
    for side, rt in rt_sides:
        if isinstance(rt.port, bool) or not isinstance(rt.port, int) or not 1 <= rt.port <= 65535:
            raise ValueError(f"runtime.{side}.port must be a port number, got {rt.port!r}")
    # top_logprobs_num: int >= 1 or the literal "auto"
    tln = cfg.scoring.top_logprobs_num
    if not (tln == "auto" or (isinstance(tln, int) and not isinstance(tln, bool) and tln >= 1)):
        raise ValueError(
            f"scoring.top_logprobs_num must be a positive integer or 'auto', got {tln!r}"
        )
    for name, val, lo in (
        ("prompts.n", cfg.prompts.n, 1),
        ("prompts.max_prompt_tokens", cfg.prompts.max_prompt_tokens, 1),
        ("generation.max_new_tokens", cfg.generation.max_new_tokens, 1),
        # 0 would build asyncio.Semaphore(0) and hang every request forever
        ("scoring.concurrency", cfg.scoring.concurrency, 1),
        ("scoring.score_chunk_size", cfg.scoring.score_chunk_size, 0),
        ("metrics.bucket_size", cfg.metrics.bucket_size, 1),
        ("metrics.worst_n", cfg.metrics.worst_n, 0),
    ):
        if isinstance(val, bool) or not isinstance(val, int) or val < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {val!r}")
    if not isinstance(cfg.generation.temperature, (int, float)) or cfg.generation.temperature < 0:
        raise ValueError(f"generation.temperature must be >= 0, got {cfg.generation.temperature!r}")
    if not isinstance(cfg.generation.top_p, (int, float)) or not 0 < cfg.generation.top_p <= 1:
        raise ValueError(f"generation.top_p must be in (0, 1], got {cfg.generation.top_p!r}")

    if not cfg.allow_generations:
        if cfg.ingest is None or not cfg.ingest.source:
            raise ValueError(
                "allow_generations: false requires an 'ingest' section with a 'source'"
            )
        if cfg.ingest.format not in INGEST_FORMATS:
            raise ValueError(
                f"ingest.format must be one of {INGEST_FORMATS}, got {cfg.ingest.format!r}"
            )

    # Trips detection errors early, and enforces one engine per run: every
    # candidate must match the reference engine (mixed pairs unsupported).
    for cand in cfg.candidates:
        if cand.engine is not cfg.reference.engine:
            raise ValueError(
                f"reference and candidate {cand.name!r} must use the same engine: "
                f"{cfg.reference.image!r} -> {cfg.reference.engine.value}, "
                f"{cand.image!r} -> {cand.engine.value}"
            )

    # Port/GPU sanity across the managed servers.
    managed_ports = {}
    if cfg.runtime.ref.managed:
        managed_ports["ref"] = cfg.runtime.ref.port
    for name, rt in cfg.runtime.candidates.items():
        if rt.managed:
            if rt.port in managed_ports.values():
                # ref + candidates never run at once (candidates are sequential), so a
                # shared port is fine; only a duplicate among concurrently-up servers matters.
                pass
            managed_ports[name] = rt.port
        ref_gpus = set(cfg.runtime.ref.gpus)
        if cfg.runtime.ref.managed and rt.managed and ref_gpus & set(rt.gpus):
            print(
                f"warning: reference and candidate {name!r} GPU sets overlap "
                f"({sorted(ref_gpus & set(rt.gpus))}) — fine for sequential staging "
                f"(servers up --only …), but they cannot run concurrently",
                file=sys.stderr,
            )
    return cfg
