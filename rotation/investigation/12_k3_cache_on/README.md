# Kimi-K3 GPQA: cache-ON headline + BF16 control at the same 168-question prefix

Two things were open on the K3 row. (1) Its INT2 headline was the only cache-OFF
number in a ten-model sweep whose other nine rows are cache-ON. (2) Its BF16
control was n=14, far too thin to be the paired column in the final table.

Everything below is scored on the **contiguous gap-free prefix** of the seed-0
GPQA-Diamond shuffle (`k3_prefix_score.py`, `pos == 0..n-1`). The harness's own
`running_score` reads ~12 pp high mid-run purely from length survivorship — fast
questions return first, cap-hitting loops land last — so positions with a gap
before them are excluded, always.

`k3_align.py` verifies that two runs are pairable before any McNemar: every
shared `pos` must agree on `orig_idx` **and** on the correct letter, because
`pos` indexes a seed-0 shuffle and the option permutation is a second seed
(`10000 + orig_idx`). All four K3 result files pass with 0 mismatches.

## Table-ready rows

| arm | prefix cache | n | score | answered | capped (`finish_reason=length`) | loop rate | cache hit rate | completion tokens (med / mean / p90) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **BF16 KV control** | OFF | **168** | **107/168 = 63.7 % ± 3.7** | 125 (74.4 %) | 64 (38.1 %) | 22.6 % | 0.00 % | 5 128 / 13 567 / 30 720 |
| BF16 KV control (full file) | OFF | 198 | 127/198 = 64.1 % ± 3.4 | 148 (74.7 %) | 77 (38.9 %) | 24.2 % | 0.00 % | 5 247 / 13 789 / 30 720 |
| INT2 OSCAR (existing headline) | OFF | 168 | 93/168 = 55.4 % ± 3.8 | 111 (66.1 %) | 56 (33.3 %) | 12.5 % | 0.00 % | 5 733 / 13 121 / 30 720 |
| INT2 OSCAR **cache-ON** | ON | **47** *(target ≥168, in flight)* | 25/47 = 53.2 % ± 7.3 | 33 (70.2 %) | 13 (27.7 %) | 6.4 % | **9.56 % cumulative** | 6 027 / 11 905 / 30 720 |
| INT2 OSCAR, same allocation as BF16 | OFF | 143 | 80/143 = 55.9 % ± 4.2 | 106 (74.1 %) | 76 (53.1 %) | 42.0 % | 0.00 % | 30 720 / 18 195 / 30 720 |

### Paired deltas (McNemar, exact two-sided sign test on discordant pairs)

| pair | n | delta (BF16 − INT2) | discordant | p |
|---|---:|---:|---:|---:|
| BF16 vs INT2 cache-OFF (headline) | 168 | **+8.3 pp** (paired SE 4.0) | 30 / 16 | **0.054** |
| ... dropping the 14 positions served by the retracting 0.72 BF16 server | 154 | +9.1 pp (SE 4.0) | 27 / 13 | 0.039 |
| ... those 14 positions alone | 14 | +0.0 pp | 3 / 3 | 1.000 |
| BF16 vs INT2 cache-OFF, **same 16-GPU allocation** | 143 | +7.0 pp (SE 4.2) | 23 / 13 | 0.132 |
| BF16 vs INT2 cache-ON (thin) | 47 | +17.0 pp (SE 8.1) | 12 / 4 | 0.077 |

The delta sits exactly on the 0.05 boundary: p = 0.054 on the headline scope,
0.039 once the 14 records from the retracting BF16 server are dropped, 0.132 on
the independent same-allocation replication at n=143. Direction is consistent
across all three (BF16 ahead by 7-9 pp); significance is marginal and one seed
per arm is not enough to call it. Given that the two INT2 replications land 0.5 pp
apart on score while their failure profiles diverge wildly (next section), the
score is the stable quantity and the honest reading is **INT2 ≈ 8 pp under BF16,
p ≈ 0.05, single seed**.

## Verified config, read from the live server (`/get_server_info`), both ranks

Every arm was diffed rank 0 vs rank 1 on `kv_cache_dtype`,
`disable_radix_cache`, `mamba_scheduler_strategy`, `page_size`,
`kv_cache_quant_group_size` and `max_total_num_tokens`: **all SAME on all arms.**
This is not ceremony — on 2026-08-22 a stale handshake file brought up rank 0 as
BF16 and rank 1 as INT2, and because sglang gives every PP stage its own KV pool
it served that model silently: 47 layers in one precision, 46 in the other, no
error, a clean-looking and entirely wrong number.

| | INT2 cache-OFF (headline) | INT2 cache-ON | BF16 control |
|---|---|---|---|
| `kv_cache_dtype` | `int2` | `int2` | `bfloat16` |
| `disable_radix_cache` | `True` | **`False`** | `True` |
| `mamba_scheduler_strategy` | `no_buffer` | **`extra_buffer`** | `no_buffer` |
| `page_size` | 8 | 8 | 8 |
| `kv_cache_quant_group_size` | `None` | `None` | `None` |
| sink / recent / `recent_ring` | 64 / 256 / **263** | 64 / 256 / **263** | n/a |
| `max_total_num_tokens` | 483 592 | **264 168** | 195 672 (0.80) · 113 760 (0.72) |
| `mem_fraction_static` | 0.68 | 0.68 | 0.80 for 184/198 records, 0.72 for 14 |
| cuda graph | on, captured `[1..8]`, **0** decode steps outside a graph | same | same |
| decode-batch histogram (`#running-req`, PP=2 microbatches) | 1:638 2:5147 3:7017 4:5830 | 1:17 2:2224 3:1234 4:1963 | 1:407 2:3066 3:3022 4:1957 |
| retractions | 0 | 0 | 0 — **except the 0.72 server: 32** |
| client concurrency | 6 | 6 | 6 |

`recent_ring = 263 = 256 + N_Q − 1` on both INT2 arms, as required.
`kv_cache_quant_group_size` is correctly unset: K head dim 192 = 3×64, so any
group size dividing both 192 and 128 yields a group count with a factor of 3 that
no INT2 write path supports.

`extra_buffer` costs **45 %** of the KV pool (264 168 vs 483 592 tokens at the
same 0.68). It still clears the 202 752 tokens that 6 × (30 720 + prompt) needs,
so the cache-ON arm holds matched concurrency without a mem-fraction bump — the
BF16 arm is the one that needed 0.80 to get there.

## The one asymmetry that exists, and its measured size

14 of the 198 BF16 records (positions 0-9, 11, 12, 13, 15) were served by a
mem-fraction-0.72 server whose pool was 113 760 tokens — below the 202 752 the
matched concurrency needs — and that server **retracted 32 times**. That is
exactly the queue-instead-of-run asymmetry that produced MiniMax-M2.7's
fictitious −21 pp, so it was measured rather than argued about: dropping those 14
positions moves the delta from +8.3 to **+9.1** pp, i.e. the retraction slightly
*depressed* BF16 rather than flattering it, and on those 14 positions the two
arms are exactly tied (8/14 each). The other 184 records came from 0.80 servers
with 0 retractions.

## The failure-mode decomposition does NOT replicate — retract two "known facts"

`k3_failure_report.py` was run on **two independent INT2 cache-OFF replications**
at byte-identical config (same args, same OSCAR env, same 0.68 / 483 592 pool,
same THREADS=6, same 30 720 cap, temp 1.0 / top_p 0.95) and on the BF16 arm:

| | BF16 (n=168) | INT2 rep A = gpqa9 (n=168) | INT2 rep B = ctl1 (n=143) |
|---|---:|---:|---:|
| **score** | **63.7 %** | **55.4 %** | **55.9 %** |
| answered | 74.4 % | 66.1 % | 74.1 % |
| capped | 38.1 % | 33.3 % | **53.1 %** |
| capped **inside** the think block | 17.9 % | **31.5 %** | 21.0 % |
| of capped: closed think first | 34/64 = 53 % | **3/56 = 5 %** | **46/76 = 61 %** |
| score in the 30 719-30 720 bucket | 37.5 % | **1.8 %** | **42.1 %** |
| accuracy among answered (diagnostic) | 85.6 % | 83.8 % | 75.5 % |
| tail is a verbatim repetition loop | 47/64 = 73 % | 34/56 = 61 % | 61/76 = 80 % |

**Only the score replicates** (55.4 vs 55.9, and both ~8 pp under BF16). Every
failure-mode statistic moves violently between two runs of the same
configuration, and BF16's profile sits *between* the two INT2 replications on
every failure axis. Two premises carried into this task therefore do not survive
replication and should be retracted:

* *"100 % of INT2 non-termination occurs inside the think block (53 of 56 capped
  never emitted `<|close|>think`)"* — replication B gives **46 of 76 capped
  responses closing think first**, i.e. 39 % inside-think, not 95 %.
* *"score-vs-length is a cliff, 91.7 % below 2K tokens and 1.8 % at the cap"* —
  the monotone decline with length is robust (92.0 % below 2K in B), but the
  depth at the cap is not: **42.1 %** in B versus 1.8 % in A. A capped response
  is not automatically a zero; it often already contains an extractable answer.

The practical consequence: at n≈150 with a third to a half of questions hitting
the cap, K3's failure decomposition has run-to-run variance larger than the
BF16-vs-INT2 difference on the same axis, so no single-run failure-mode
attribution about this model should be trusted, and cap-rate deltas between arms
need several seeds before they mean anything. What survives is the coarser and
already-established point: the loss is concentrated in non-termination rather
than in wrong answers, and the BF16 arm truncates at least as much as INT2
(38.1 % vs 33.3 % / 53.1 %), so non-termination is not caused by 2-bit KV. That
remains consistent with `chat_template = None` (confirmed on the live server)
being the leading explanation.

The only known difference between the two INT2 replications is the weight
*source*: A read the original `moonshotai/Kimi-K3` snapshot (deleted from the HF
PVC on 2026-08-21 ~21:45 UTC), B reads `/shared/kimi-k3-base`, the MTP-stripped
view of Together's mirror. Both come up at exactly 483 592 KV tokens, and both
score ~55.5 %, so this is very unlikely to be the cause — but it is the one
uncontrolled variable and is recorded rather than assumed away.

## Why response-section-only extraction is load-bearing here

`k3_gpqa.py` scores the `<|open|>response<|sep|>` section and keeps a
whole-string reading as `score_fullraw`. At n=168 the whole-string regex reads
**BF16 70.8 %** and **INT2 58.9 %** against the correct 63.7 % / 55.4 % — it
inflates BF16 by 7.1 pp and INT2 by 3.5 pp, so it would also inflate the *paired
delta* from +8.3 to **+11.9 pp**. The inflation is asymmetric because BF16 emits
more mid-thought `Answer: X` candidates that it never commits to (it closes the
think block far more often). A whole-string regex is therefore not a harmless
convenience on this model; it manufactures a third of the gap.

## Cache hit rate — correcting the 0.2 % figure

0.2 % (64 cached / 29 968 new) was the **gate probe's** number, measured on a
freshly started server that had only ever seen the probe's own unique prompts.
Over the cache-ON server's whole life the scheduler reported **4 760 cached /
45 040 new = 9.56 %** of prompt tokens; net of the gate's own ~30 k uncached
probe traffic, the scoring run itself ran at ≈24 % of prompt tokens cached
(GPQA prompts share the system message plus the instruction preamble).

That is still immaterial to quality, but for a different reason than "the cache
does nothing": prompts are ~290 tokens against a ~13 500-token mean end context,
so ~100 cached tokens per question is **~0.7 % of the KV traffic**. The cache-OFF
arm measured exactly 0.00 % (0 cached / 65 504 new), so the two modes really do
differ — the point of the cache-ON run is 口径 consistency with the other nine
models, not a performance claim.

## The gate fix

`k3_gate.py` v1 certified reuse from a probe that requested **input logprobs**.
Requesting input logprobs makes sglang recompute the prompt, which suppresses
prefix-cache reuse, so the probe always read `cached_tokens=0`. v1 therefore
could not distinguish "the cache is broken" from "my probe disabled the cache":
on run gpqa9 it first **FAILED** the arm and downgraded the whole run to
cache-OFF (that is the 168-question headline), and a later incarnation reported
INCONCLUSIVE and kept cache-ON (that is the 47-question file). One logical run,
two server configurations, two result files.

v2:

* **Never** uses input logprobs. The probe is a plain generation request — the
  same request shape the scoring run sends — so whatever reuse the scoring run
  gets is what the probe measures.
* Reads reuse from the server's own counters: per-request
  `meta_info.cached_tokens` plus the scheduler's cumulative `#cached-token` /
  `#new-token`. (`/metrics` is unavailable without `--enable-metrics`, and
  `/get_server_info` on this build is a flat `server_args` dump with no live
  counters, so the server log is the authoritative cumulative source.)
* Treats inconclusive as **exit 2, non-fatal** — a 30-minute weight load must not
  be discarded because a probe could not prove something.
* Refuses to **label** an arm cache-ON unless the live server reports
  `disable_radix_cache=False` **and** a non-zero hit rate accumulated. The last
  line is machine-readable (`GATE_VERDICT label=... reuse=... hit_rate=...
  correctness=... rc=...`) and `head4.sh` re-derives the final label from the
  hit rate measured over the *scored* run, not from intent.
* Checks correctness only once reuse is proven, and self-calibrates: two cold
  greedy generations set the reproducibility floor, and the warm generation must
  sit at it. A broken mamba/conv-state cache does not crash — it silently answers
  from a recurrent state captured at the wrong chunk boundary — so this is the
  only place it can be caught.

Smoke-tested against the live cache-OFF server: it correctly refuses the label
and exits 2 without sending a single generation request.

```
[gate] live server: disable_radix_cache=True mamba_scheduler_strategy=no_buffer page_size=8 kv_cache_dtype=int2 kv_cache_quant_group_size=None
[gate] server reports disable_radix_cache=True: this arm is cache-OFF and must not be labelled cache-ON
GATE_VERDICT label=cache-off reuse=no hit_rate=0.000% correctness=n/a rc=2
```

## Files

| file | role |
|---|---|
| `k3_gate.py` | the fixed gate (v2) |
| `k3_args4.sh` | launch args; prefix-cache mode is part of the arm name (`int2on`/`int2`/`bf16on`/`bf16`) so a results file can never blend two server configs |
| `head4.sh` / `worker4.sh` | two-rank driver: worker-alive barrier, RUNTOKEN-stamped handshake, rank-config diff, gate, per-arm labelling |
| `k3_align.py` | proves two runs are pairable before any McNemar |
| `k3_prefix_score.py` | contiguous-prefix scorer + matched-scope comparison |
| `k3_failure_report.py` | termination-class breakdown |
| `../../../../k8s/zz-k3-on.yaml` | the cache-ON job pair (`zz-k3-onh` / `zz-k3-onw`, RUN=`on1`) |
| `../../../../k8s/zz_k3_on_launcher.sh` | bounded resubmission watchdog for that pair |
| `../../../../k8s/zz_k3_handoff.sh` | hands the 16-GPU allocation from `ctl1` to the cache-ON pair |

Results live on the `shared-data` PVC:
`/shared/kimi3_out/ctl1/gpqa/results.{bf16,int2}.jsonl`,
`/shared/kimi3_out/gpqa9/gpqa/results.{off,on}.jsonl`,
`/shared/kimi3_out/on1/gpqa/results.int2on.jsonl`.

## Not reached

The cache-ON arm needs **≥168 contiguous positions**. K3 answers a GPQA question
every 174-244 s at 6-way concurrency (63-69 aggregate tok/s against a 30 720-token
cap that a third to a half of questions hit; the spread is the cap-rate variance
above), so 168 positions is **8-11.5 h of dispatch** — more than one arm's
deadline, which is why `ARMS="int2on int2on int2on"` and the
skip-positions-already-in-the-JSONL resume path matter.

Why the existing cache-ON file stopped at 47: its worker Job (`zz-k3-w9`) failed
out from under a healthy head ~8 700 s into the run, at 50 of 198 completed. That
is the failure the watchdog exists for — it now resubmits and the next arm resumes
the same file.

State at hand-off: `zz-k3-onh`/`zz-k3-onw` queued (`wait_capacity` /
`blocked_by_head`), watchdog live (`k8s/launcher_on.pid`), n=47. **Until it passes
168 the cache-ON row is not table-ready**, and the cache-OFF 55.4 % must keep
carrying the headline with the cache mode stated next to it.

The BF16 control is cache-**OFF**. Pairing it against a cache-ON INT2 arm leaves
one uncontrolled variable, immaterial in size (≈0.7 % of KV traffic) but real; the
fully matched pair is BF16 vs INT2 cache-OFF. A cache-ON BF16 arm was not run —
it would cost another 8-11 h of the same 16 GPUs, and on a hybrid
linear-attention model `extra_buffer` and `--disable-radix-cache` are mutually
exclusive, so no single arm can be both.
