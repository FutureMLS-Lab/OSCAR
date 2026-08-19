# CLAUDE.md — kl-probe

Measures how close a quantized model (e.g. NVFP4) is to a higher-precision
reference (MXFP8/bf16) at the **token-distribution level**, driven through the
real SGLang/vLLM serving stack.

## Method

Teacher-forced next-token comparison. Two input modes (`allow_generations`):

- **generate** (`allow_generations: true`): sample prompts → generate a
  completion on the *reference* server.
- **ingest** (`allow_generations: false`): the reference completions already
  exist (e.g. an HLE response cache) → assemble them; **no generation**.

Then, for the fixed `prompt + generation` token ids: prefill-score them on the
reference and on every candidate, and compare the per-position next-token
distributions. Both sides go through the identical prefill path, so the only
variable is the checkpoint.

One `reference`, a **list** of `candidates` (a quantization ladder). The
reference is scored **once** and reused; each candidate is scored once and
compared to it. Scoring covers **all positions** (from index 1), not just a
generated suffix.

## Run

`uv` lives at `~/.local/bin` (not on PATH). Docker needs `sudo` (passwordless).

```bash
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/data/huggingface
uv sync                                   # first time
uv run probe run -c configs/<cfg>.yaml --down          # full pipeline
uv run probe run -c <cfg> --null --down                # ref vs itself (sanity: KL≈0)
# smoke: use a config with small prompts.n / generation.max_new_tokens
# stages: prompts | ingest | generate | score --model ref|<candname> | metrics
# servers: uv run probe servers up|down|logs -c <cfg> [--only ref|cand] [--cand NAME]
```

`probe run` stages sequentially: **ingest/generate → score ref (server up) → ref
down → per candidate {up → score → down} → metrics vs ref → ladder table**.
Candidates share one GPU set through the single reusable `cand` server slot.

Artifacts land in `runs/<run_name>/<datetime>/` — every invocation gets its own
timestamped subdir, so runs never overwrite each other: `generations.jsonl`,
`scores_ref.parquet`, `scores_<cand>.parquet`, `tokens_<cand>.parquet`,
`metrics.json` (keyed by candidate, with per-bucket metrics + `worst_positions`).
Because each run is a fresh dir, artifacts are always recomputed from scratch;
there is no cross-run reuse.

## Config (see `configs/*.yaml`)

- `reference` + `candidates: [...]` (each candidate needs a unique `name`).
  A single-candidate run is just a one-element `candidates` list.
- `runtime.ref` + `runtime.candidates` (map name → {gpus, port, base_url}).
- `allow_generations`, `ingest` (source/format/caps), `generation`, `scoring`
  (`top_logprobs_num: auto` derives from the reference `generation_config` top_k,
  fallback 40; `concurrency`; `score_chunk_size` reserved), `metrics.bucket_size`.
- Engine auto-detected from the image name (`sglang`/`tgl`/`vllm`); all servers
  in a run must resolve to the same engine and share a tokenizer.

## Layout

`src/kl_probe/`: `config.py` (YAML→dataclasses), `ingest.py` (precomputed-gen
assembly), `prompts.py`/`generate.py` (generate mode), `score.py` (prefill
scoring, streaming parquet), `metrics.py` (KL + per-bucket + worst_positions),
`servers.py` (docker orchestration), `{sglang,vllm}_client.py`, `cli.py`.
`docker/docker-compose.yml` (2 slots: ref + cand), `patches/` (loader patches),
`configs/`, `tests/`, `docs/precomputed-generations.md` (ingest-mode design).

## Dev

```bash
uv run pytest -q            # tests
uv run ruff check .         # lint  (config in pyproject.toml [tool.ruff])
uv run ruff format .        # format
```

## Serving notes / gotchas

- MiniMax-M3 (~428B) needs the tgl ECR image + TP=4; not in stock sglang.
- `prewarm` reads the checkpoint's safetensors into page cache before mmap (WekaFS
  cold-load is otherwise hours). Loads take ~10 min.
- Compose is `restart: "no"` on purpose — a failed start must not restart-storm.
- **Scoring is memory-heavy**: all-position input-logprobs on long (~2.5k-token)
  sequences at high concurrency OOMs the MoE forward. Keep `scoring.concurrency`
  low (1–2) and `--mem-fraction-static` modest (~0.75) for ingest runs; scoring is
  prefill-only so it needs little KV cache.
- If servers fail to start with `CUDA error: ...devices busy or unavailable` on
  idle GPUs, the host GPU stack is wedged — check `torch.cuda.set_device(0)` at
  **bare metal** first (it fails there too); needs `nvidia-smi --gpu-reset` /
  fabric-manager restart / reboot, not a code change.
</content>
