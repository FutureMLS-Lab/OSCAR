# Model verification harness

Answers one question per model: **does it serve correctly with prefix cache and
CUDA graph both on?** Not "what does it score" — scores come from
`rotation/run/<model>.sh`, which drives the real evals.

```
all.sh        the model table + sweep driver; `all.sh <name>` runs just one
mha.sh        INT2 KV path  (Qwen3*, Qwen3.5*, Gemma-4, MiniMax-M2.7/M3)
mla.sh        packed 2-bit MLA latent path (GLM-5.2/5.3, Kimi-K3)
_common.sh    probe, garbling judge, verdict — shared by both launchers
```

## The four criteria

| criterion | how | why it is measured this way |
|---|---|---|
| CUDA graph | capture lines in the server log | absence is otherwise silent |
| prefix cache | the **same long** prompt sent twice must report `#cached-token` | a **short** probe reports 0 and reads as "cache off" when the prefix is merely below block granularity |
| not garbled | four **shape** checks on the output | a letter-ratio judge gave four false verdicts — error strings are mostly letters, so they passed |
| 0 traceback | counted **before** teardown | the gloo shutdown path emits its own errors and once failed a healthy 30B-A3B |

`CLEAN` means **did not collapse**, not **answered correctly** — a fluent,
off-task answer passes. K3's latent BF16 arm was fluent and entirely off-task.

## Things that bit us

- **Rotations are discovered by glob, not by name.** MiniMax-M3 ships
  `k_rotation_hadamard.pt`; a hardcoded `qqt_r_h_pbr` reported "no rotation",
  which reads as "model unsupported" when the files were present.
- **GPUs are drained before sizing.** A leftover allocation is charged to the
  next model — it once made a 12B report 79 GiB in use.
- **MLA pins `--kv-cache-dtype bfloat16`.** sglang picks `fp8_e4m3` for a DSA
  model on SM100+, which puts ~28% of every 512×512 rotation subnormal; that arm
  scored 5.56.
- **Each MLA model gets its own `HOME`/`TRITON_CACHE_DIR`.** Two GLM jobs sharing
  one DeepGEMM cache put one at 2.22 s/it (a 20-hour warmup) while the other ran
  at 9800 it/s.
- **Verdicts append to a file on the volume after each model.** exec stdout has
  lost results twice on pod death.
- **`KEEP_WEIGHTS=1` by default here.** The purge exists because the pod was once
  evicted for ephemeral-storage overrun; on a large volume it would instead
  delete 1.5 TB and force a re-download before every model.

## Run

Run it on one 8-GPU host with the weights reachable through `HF_HOME`:

```bash
export OUT=/path/for/per-model-output V=/path/to/verdicts.txt
bash rotation/verify/all.sh            # every model, resumes from $V
bash rotation/verify/all.sh qwen3-8b   # just one
```

How you get that host is deliberately not described here. Every verdict is
appended to `$V` as it is earned, so an interrupted sweep resumes instead of
starting over -- which is the only part of the scheduling story that belongs
in this repo.

## The K3 port that had to be rolled back

Between 09-12 and 09-15 K3's model file was replaced with upstream's wholesale,
on the theory that our reimplementation was the reason K3 trailed stock sglang
(70.83 vs 95.83 on the same 48 questions). The ported file loaded all 96 shards,
captured CUDA graphs on both pipeline stages, decoded at 241 tok/s -- and
emitted word salad. Thirty-eight commits went into repairing its fallout before
the branch was rolled back to the file that scored 70.83.

What made it expensive was diagnosing forwards instead of backwards. The
evidence that the port was the regression was already on disk the whole time:
a 70.83 run dated 09-11 00:05, one minute before the commit whose message
records it. Bisecting our own commits would have found that in minutes.

Three mistakes worth not repeating:

- **A non-control used as a control.** `run/kimi-k3.sh` exported
  `SGLANG_OSCAR_K3_MLA_LATENT=1` unconditionally, so the "BF16 arm" still ran
  fork-local latent code. Several rounds of "the KV path is not implicated"
  rested on it. The flag is overridable now.
- **A cause declared before the path was proven.** `gate_up_interleaved` was
  threaded through FusedMoE and never read, and the with-bias reader disagrees
  with the loader's layout -- a real defect, fixed and now guarded by
  `audit_dead_params.py`. But K3's experts carry no bias, so K3 never ran that
  code. Matching the shape of a symptom is not evidence.
- **Length compared across different parsers.** Upstream's median 536
  characters is `content` with thinking split out by `--reasoning-parser
  kimi_k3`, which this fork does not have. Our figures included the thinking.

Two fixes from the attempt were kept because they are real and live in shared
code: the KDA decode metadata use-after-free (`_decode_query_start_loc`) and the
`gate_up_interleaved` reader. Both re-ran clean through the full sweep.
