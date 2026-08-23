# Pre-merge regression probe contract

Every model row must be produced the same way or the rows are not comparable.
Read this whole file before launching anything.

## Repo / tree

* Source of truth: `/home/admin/CoQuant/.RUD/hybridmodel-testing/work/CoQuant`,
  branch `zhongzhu/hybrid-model`. Remote is **`oscar`**, not `origin`.
* On charlie the deployable tree is `/shared/zz-m27-recheck/tree_post`.
* **Never import Python from a shared tree.** Other jobs `git checkout -f` into
  `/shared/...` while your server imports from it; two arms then silently
  measure different code. Copy to pod-local `/tmp/tree` and pin, exactly as
  `investigation/10_m27_graphbs/probe_job.yaml.tmpl` does
  (`ISOLATE_TREE=1` in `run_arm10.sh` already does this).
* `third_party/simple_evals` is a **gitlink**. A plain `git clone` leaves it
  empty and it fails only *after* the 10-40 minute weight load. Copy it from the
  source tree and assert the files exist before launching.
* Record `git -C $TREE rev-parse HEAD` in your output. All rows must share one SHA.

## charlie hygiene (non-negotiable, each of these has cost a run)

```
export HOME=/tmp/h XDG_CACHE_HOME=/tmp/xdg TMPDIR=/tmp
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext TVM_FFI_CACHE_DIR=/tmp/tvmffi
export SGLANG_CACHE_DIR=/tmp/sglcache TRITON_CACHE_DIR=/tmp/triton
export HF_HOME=/shared/huggingface
export PYTHONPATH=/shared/charlie/py_extra     # jsonschema + tiktoken overlay
```

* `/home/charlie` PVC is **100% full / 0 bytes free**. Anything defaulting under
  `$HOME` dies with ENOSPC.
* The shared `oscar` conda env lacks `jsonschema` and `tiktoken`; the
  `/shared/charlie/py_extra` overlay on `PYTHONPATH` supplies them.
* Exclude nodes `research-common-h100-117` (corrupts INT2 output) and
  `research-common-h100-073` (driver-broken) via nodeAffinity `NotIn`.
* Nodes `046/055/080/090` advertise only 7 GPUs, so an 8-GPU pin pends forever.
* The harness reads **`MEM_FRAC`**, not `MEM_FRACTION_STATIC`.
* Delete pods by **exact name**. **NEVER** use
  `--field-selector status.phase=Failed` -- that already destroyed another
  user's pod here.
* Leave `zz-k3-*` and `zz-glm52-*` alone; they belong to other work.
* Always pass `kubectl --context charlie` (or `--context b200`). The
  current-context auto-flips and a wrong context mimics a namespace purge.

## Fixed probe protocol -- do not vary these

| knob | value | why |
|---|---|---|
| task | GPQA-diamond, `NUM_EXAMPLES=24` | enough real samples; long gens reach the decode flush |
| `MAX_NEW_TOKENS` | 8192 | must exceed the HP-recent ring (256/512) or the decode flush never fires and the probe proves less than it looks |
| `MAX_RUNNING` | 32 | makes the captured set `[1,2,4,8,12,16,24,32]` |
| **client concurrency** | **16** | held identical across every model and recorded next to every number |
| temperature / top_p / top_k | 1.0 / 0.95 / 40 | never probe at temp 0: greedy makes every INT2 config emit identical whitespace soup |
| radix cache | **ON** | we are verifying it works, not disabling it |
| CUDA graph | ON, `CUDA_GRAPH_MAX_BS=32` | never leave it pinned to 1: that turns graphs *off* above batch 1 |
| `ABSORB_V` | 0 (except Gemma-4, which hardcodes 1) | |

`MAX_RUNNING` and client concurrency **must stay decoupled**:
`get_batch_sizes_to_capture` truncates the captured set to
`max-running-requests` *and appends it*, so `--max-running-requests 16` would
make 16 a captured size and blind the probe.

Client concurrency is an **experimental variable**, not a throughput knob -- it
selects which captured graph replays. M2.7 produced two opposite fake results
(+2.0 and -21.05) purely from mismatched concurrency. Record it every time.

## The three things to report, and what counts as evidence

### 1. Coherent English

Run the shared judge on the run's `io_log.jsonl`:

```
python3 rotation/investigation/11_premerge_regression/coherence_judge.py \
    $RUN_DIR/io_log.jsonl --show 2
```

It gates on digit-soup (digit fraction **and** word sparsity) and repeated
n-grams with a `top_count >= 2` gate, plus a char-level loop check. It does
**not** use a letter ratio or alpha density -- those produced false verdicts
here in both directions. Paste one real generated excerpt into your report. A
terse correct answer is not garbage.

### 2. CUDA graph actually active

"graph enabled in the flags" is **not** evidence. M2.7 ran
`--cuda-graph-max-bs 1` for months, which means graphs are off above batch 1.
Report, from the live `server.log`:

* the captured set (`Capture cuda graph bs [...]`)
* `decode_bs_hist` -- the histogram of `#running-req` over decode batches
* the fraction of decode steps with `cuda graph: True`

`investigation/06_m27_recheck/analyze.py $RUN_DIR --arm <name> --json` computes
all of these. Note `padded_replay_fraction` must be counted as
`graph == True AND bs not in captured` -- a non-captured size with
`cuda graph: False` is eager and pads nothing. Counting non-captured sizes alone
once reported 99.5% padded where the truth was 0%.

### 3. Prefix cache actually on

Read `disable_radix_cache=` back **from the live server log**, not from the
script defaults, and report a non-zero hit rate (needs `--enable-cache-report`).
On GPQA the ceiling is ~17% because the 198 prompts share only a ~56-token
instruction prefix: **~17% is success, near-zero means the cache was off.**
With `NUM_EXAMPLES=24` the reuse surface is smaller, so expect ~10-17%.

Mamba / linear-attn models (Qwen3.5-4B, Qwen3.5-35B-A3B) need
`--mamba-scheduler-strategy extra_buffer` for prefix caching, and that flag
**raises ValueError together with `--disable-radix-cache`** -- so a cache-OFF
control must use `no_buffer`.

## Auditor caveat

The mixed-KV auditor's damage counter is weak: it reads zero even on known-broken
code, and its `DECODE_WRITE_INTO_HP_PREFIX` budget (8 per kind per process) is
entirely burned during graph-capture warmup. The reliable discriminator is
allocator-level: `hp_prefix_page0_ever_allocated == false` and minimum allocated
page == 1. Report those two if you enable `SGLANG_MIXED_KV_AUDIT=1`.

## Reporting

Report one row: model, TP, SHA, coherent-English verdict + a real excerpt,
captured set + decode_bs_hist + graph-true fraction, `disable_radix_cache` +
hit rate, client concurrency, and pass/fail. **Say plainly what you could not
verify** -- a short honest row beats a complete-looking one with guesses in it.
