# kl-probe

Measures how close a quantized model (e.g. NVFP4) is to its reference
(bf16/FP8) at the token-distribution level, through the actual SGLang serving
stack.

**Method** (teacher-forced comparison): sample prompts → generate with the
reference model → prefill the identical `prompt+generation` token ids into
*both* servers → compare the next-token distribution at every generated
position. Both distributions come from the same prefill code path, so the only
variable is the checkpoint.

**Metrics**: per-token KL(ref‖cand) (headline; both distributions renormalized
over ref's top-k support, so it is a true non-negative KL between the top-k
conditionals — uncovered tail mass is reported), top-1 agreement, top-5/top-k
overlap, exact Δlogprob on the actual token, plus breakdowns by position and by
reference entropy, and a worst-N drill-down with decoded context. Diagnostics
include a prefill-vs-decode self-consistency floor for the reference model.

## Setup

```bash
cd /data/sgambhira/kl-probe
uv sync
export HF_HOME=/data/huggingface     # shared cache on this box
export HF_TOKEN=...                  # if the models need it
```

## Usage

```bash
# End-to-end: brings up both servers via sudo docker compose (GPUs from config),
# waits for health, runs prompts -> generate -> score x2 -> metrics -> report.
uv run probe run -c configs/qwen3.6-27b.yaml

# Sanity check the pipeline: reference vs itself => KL ~ 0, top-1 ~ 100%.
uv run probe run -c configs/qwen3.6-27b.yaml --null

# Quick smoke before a full run: use a config with small prompts.n /
# generation.max_new_tokens (run sizes come from the config).
uv run probe run -c configs/qwen3.6-27b.yaml

# Manage servers / individual stages
uv run probe servers up|down|logs -c configs/qwen3.6-27b.yaml
uv run probe prompts|generate|metrics|report -c ...
uv run probe score --model ref|cand -c ...
```

Artifacts land in `runs/<run_name>/<datetime>/` — each invocation gets its own
timestamped subdir, so runs never overwrite each other: `prompts.jsonl`,
`generations.jsonl`, `scores_{ref,cand}.parquet`, `tokens.parquet` (per-position
metrics), `metrics.json`.

## Sizing

- Default 512 prompts × ≤256 tokens ≈ ~100–130k compared positions — mean KL
  and top-1 agreement are stable well below this; p95/p99 are usable.
- For tail hunting (p99.9 / max-KL outliers) bump to ~2048 × 512 (≈0.5–1M).
- `metrics.json` carries a sequence-level bootstrap 95% CI on mean KL, so an
  undersized run is visible rather than silently noisy.

## Engines (sglang | vllm)

Each model's config carries its own docker `image:`; the serving provider is
**auto-detected from the image name**, which must contain exactly one of the
`Provider` tokens: `sglang`, `tgl`, or `vllm` (anything else is a validation
error). `tgl` is Together's internal sglang fork — it speaks the sglang API
and shares the sglang client/launcher, but is a distinct provider, so a mixed
sglang/tgl pair is rejected like any other engine mismatch (two different
builds). Both models must resolve to the same provider. `extra_args` are passed to the engine's
launcher verbatim, so use the matching flag dialect (sglang:
`--context-length`, `--mem-fraction-static`; vllm: `--max-model-len`,
`--gpu-memory-utilization`). For vllm, `--max-logprobs` is auto-appended from
`scoring.top_logprobs_num`. Design details: `docs/vllm-support.md`.

```yaml
reference:
  model_path: Qwen/Qwen3.6-27B
  image: vllm/vllm-openai:nightly     # -> engine: vllm
  extra_args: "--max-model-len 262144 --gpu-memory-utilization 0.85"
runtime:
  ref:
    gpus: [2]
    port: 38097
```

Two more per-model knobs that change what the server computes (unlike
`runtime.*` placement, which only affects where it runs):

```yaml
candidates:
  - model_path: togethercomputer/MiniMax-M3-NVFP4-skip10mxfp8-0706
    image: 598726163780.dkr.ecr.us-west-2.amazonaws.com/smg:...-tgl-...   # -> engine: tgl
    env:                                # engine knobs that have no CLI flag
      SGLANG_MINIMAX_SPARSE_DECODE_BACKEND: trtllm
      TORCHDYNAMO_DISABLE: 1
    patches:                            # loader patches (host .py paths), run
      - patches/minimax-m3-fp4-excluded-moe-mxfp8.py # in-container
```                                     # before the server starts

`env` entries are injected into the service environment; `patches` are
bind-mounted read-only and applied with `python3`, in order, before the server
process is exec'd (e.g. the excluded-MoE MXFP8 rebuild that MiniMax-M3
skip10-family checkpoints require). A failing patch crash-loops the container —
check `probe servers logs`. Local checkpoints under `/scratch` work as
`model_path`: the compose services mount `/scratch` at the same path.

**Sequential staging** — when a pair doesn't fit on the box concurrently, the
two servers may share a GPU set and run one at a time (`probe servers up/down
--only ref|cand` around the per-stage commands); overlapping `runtime.*.gpus` only
warns. See `configs/minimax-m3-wildchat.yaml` for the full recipe — it cuts
the 428B M3 pair from 6 GPUs concurrent to 4 sequential.

## Notes

- Prompts are tokenized once (reference tokenizer + chat template) and sent as
  `input_ids`; the tool refuses to run if the two tokenizers differ. Multi-turn
  dataset rows are reduced to their first user message (single-turn eval).
- KL detail: candidate probs for ref-top-k tokens missing from the candidate's
  top-k are clamped to the candidate's k-th prob, then both sides are
  renormalized over ref's top-k support. The uncovered ref tail mass is
  reported next to the headline number.
- Generation uses `sampling_seed = generation.seed + pid`, so a run is
  reproducible in intent; exact determinism still depends on sglang batching.
- Each invocation writes to a fresh `runs/<run_name>/<datetime>/`, so runs never
  clobber each other and there is no cross-run artifact reuse.
- Keep `reference.image` and `candidate.image` identical (pinned digest or
  dated tag) so the build is never the variable — a convention, not enforced.
  Docker needs `sudo` on this box; `HF_TOKEN` is passed via the environment,
  never written to disk.
