import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    SGLANG_SORT_WEIGHT_FILES = EnvBool(False)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_DISABLE_REQUEST_LOGGING = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    SGLANG_CUDA_COREDUMP_DIR = EnvStr("/tmp/sglang_cuda_coredumps")
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)


    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("multinomial")
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_ENABLE_MIXED_KV_WINDOWS = EnvBool(False)
    SGLANG_MIXED_KV_PREFIX_TOKENS = EnvInt(32)
    SGLANG_MIXED_KV_RECENT_TOKENS = EnvInt(128)
    SGLANG_MIXED_KV_HP_DTYPE = EnvStr("bfloat16")
    SGLANG_MIXED_KV_SCALE_DTYPE = EnvStr("float32")
    # Shared HP-prefix pool size (in HP slot units; rounded up to N_Q).
    # 0 = use the default of ``max_running_requests * P * 16``.
    SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS = EnvInt(0)
    # Oscar rotation + per-row clip for int2 KV cache. Learned per-layer
    # orthogonal matrices loaded from K/V rotation checkpoints.
    SGLANG_OSCAR_K_ROTATION_PATH = EnvStr("")
    SGLANG_OSCAR_V_ROTATION_PATH = EnvStr("")
    # MLA / NSA latent KV INT2 fake-quant (GLM-5.1-FP8, DeepSeek-V2 style).
    SGLANG_OSCAR_MLA_KV_ROTATION_PATH = EnvStr("")
    SGLANG_OSCAR_MLA_KV_DUMP_DIR = EnvStr("")
    SGLANG_OSCAR_MLA_KV_DUMP_MAX_TOKENS = EnvInt(8192)
    SGLANG_OSCAR_MLA_KV_GROUP_SIZE = EnvInt(128)
    # Bits per latent value in the packed MLA pool. 2 is the validated default.
    #
    # 4 exists because a model's latent may simply not be INT2-representable.
    # Measured on Kimi-K3's c_kv dump, out-of-sample relative error against the
    # 0.10 a 2-bit KV cache needs, with the shipped rotation applied:
    #
    #     2 bits, group  32   0.2255   fail
    #     2 bits, group 128   0.3491   fail
    #     3 bits, group  32   0.0957   pass, 5.00 bits/elem
    #     4 bits, group 128   0.0694   pass, 4.50 bits/elem   <- cheapest
    #
    # Four bits at group 128 beats three at group 32 because a group of 32
    # spends 2 bits/elem on scales alone. So K3's latent is worth 3.56x over
    # bf16, not the 4x that 2-bit packing would claim. GLM-5.2 is fine at 2 bits;
    # this is a property of the tensor, so measure the dump before assuming.
    #
    # Lloyd-Max is a three-threshold codebook and therefore 2-bit only; the
    # pack wrappers refuse to combine it with any other width.
    SGLANG_OSCAR_MLA_KV_BITS = EnvInt(2)
    SGLANG_OSCAR_MLA_KV_REAL_KERNEL = EnvBool(False)
    # Store the MLA/NSA latent as *packed* INT2 codes instead of fake-quantizing
    # into a BF16 cache. This is the difference between a quality measurement
    # and a deployment: with it on, ``c_kv`` occupies 2 bits/value plus group
    # params (160 B/token/layer instead of 1024 B) and ``max_total_num_tokens``
    # actually moves. ``k_pe`` stays BF16 -- MLA depends on the positional half
    # being unquantized. Requires a rotation path (there is nothing to pack
    # without one).
    SGLANG_OSCAR_MLA_KV_PACKED = EnvBool(False)
    # Group-factored packed MLA decode. Correctness is settled: exact on all
    # seven shapes of the kernel gate, and on a live DeepSeek-V2-Lite server it
    # agrees with the production kernel to reduction-order noise -- 25 of 30
    # greedy generations byte-identical over 128 tokens, mean |dlogprob| 0.0040
    # across 3325 common-prefix positions.
    #
    # Default OFF because it is a WIN ONLY AT LONG CONTEXT. Measured decode
    # throughput, group-factored / production:
    #
    #     ctx   1000   0.869x   <- 13% SLOWER
    #     ctx   4000   1.068x
    #     ctx  16000   1.249x
    #     ctx  32000   1.357x
    #
    # The crossover sits between 1k and 4k, and the reason is structural rather
    # than tunable: the path runs a second pass over the BF16 window arena, and
    # that arena is a fixed 576 tokens (prefix 64 + recent 512). At ctx=1000 the
    # packed body is ~424 tokens -- smaller than the window it pays an extra
    # kernel launch to cover.
    #
    # This cannot be made adaptive per request. Under CUDA graphs the kernel
    # choice is fixed at capture time, replay executes no Python, and sequence
    # length only arrives at replay through metadata. So it is a deployment
    # decision: turn it on for long-context serving. Moving the crossover left
    # means fusing the window pass into the packed pass to remove the launch,
    # not adding a length threshold here.
    # Launch configuration for the group-factored path's BF16 window pass.
    # It was hardcoded at BLOCK_H=16 / BLOCK_N=32 / 4 warps / 2 stages, and at
    # head_num=16 with batch=1 that BLOCK_H makes the grid (1, 1): a single CTA
    # walking all 576 window tokens on a 148-SM part. The window is a FIXED
    # cost per decode step, so it is most of why the path loses at short context
    # and wins at long -- the packed body shrinks, this does not.
    #
    # Tuned against production decode throughput at ctx 1000 / 4000, group-
    # factored over production. Original constants were BLOCK_H 16, BLOCK_N 32,
    # 4 warps, 2 stages = 0.846 / 1.052.
    #
    # Stage 1, BLOCK_H x warps (BLOCK_N 32, stages 2):
    #
    #     BLOCK_H   warps=4        warps=8
    #        16     0.846 / 1.052  0.859 / 1.078
    #         8     0.874 / 1.084  0.891 / 1.104
    #         4     0.864 / 1.087  0.889 / 1.102
    #         2     0.872 / 1.093  0.893 / 1.110
    #
    # Stage 2, BLOCK_N x stages (BLOCK_H 8, 8 warps) -- the bigger lever, and
    # the one the first sweep missed by only varying BLOCK_H and warps:
    #
    #     BLOCK_N   stages=1  stages=2  stages=3  stages=4
    #        16      0.759     0.825     0.860      --
    #        32      0.794     0.890     0.916      --
    #        64      0.841     0.934     0.935     0.918
    #       128       --       0.909     0.777     0.782
    #       256      out of shared memory: needs 898048 B, limit 232448
    #
    # BLOCK_N 64 is a genuine peak: 128 regresses (0.934 -> 0.909, and 0.777 at
    # stages 3, consistent with spilling) and 256 will not compile. Stages peak
    # at 2-3; 4 is worse.
    #
    # Net 0.846 -> 0.934 at ctx 1000 and 1.052 -> 1.150 at 4000, about +10%.
    # Run-to-run variation on this bench is ~1.7% (the 64/3 cell measured 0.952
    # once and 0.935 on repeat), so treat single-cell differences under 2% as
    # noise; the monotone trends across a row or column are the signal.
    #
    # BLOCK_H 8 rather than the marginally better 2: the bench sends ONE request
    # at a time, and a smaller BLOCK_H means more CTAs each re-loading the same
    # window rows. At batch 1 that redundancy is free because the grid is
    # otherwise empty; at batch 32 the grid is already wide and it is duplicate
    # traffic. Everything here is a batch-1 measurement.
    #
    # This does NOT make the path a default. It is still 6.6% behind production
    # at ctx 1000. Closing that needs the window pass folded into the packed
    # pass so its fixed cost disappears, which is a kernel rewrite, not a knob.
    #
    # 2026-08-27: the short-context deficit is confirmed on a second model
    # (0.877-0.967 at ctx 1000 on GLM-5.2, against 0.862 on DeepSeek-V2-Lite),
    # so it is the kernel and not node noise. gf is now a default ABOVE a
    # context threshold rather than never -- see SGLANG_OSCAR_MLA_PACKED_GF.
    #
    # And the window pass is now MEASURED as the cause, not assumed. Setting
    # both window sizes to 0 removes the arena, so gf runs ONE kernel like
    # production; gf/production at ctx 1000 then goes
    #
    #     conc 1   0.882 -> 0.984
    #     conc 8   0.914 -> 1.018
    #     conc 32  0.966 -> 0.976
    #
    # i.e. the deficit essentially vanishes at low concurrency. At conc 32 it
    # barely moves, which is the same mechanism seen from the other side: a
    # fixed per-STEP cost is already amortized by a wide batch. Two results
    # agree -- skipping fully-excluded blocks INSIDE the packed pass did
    # nothing (mean -0.011 over 9 points), because the cost was never in the
    # packed pass.
    #
    # I first wrote that the fix was to FOLD the window pass into the packed
    # pass -- one kernel, one grid. That was wrong, and it is worth recording
    # why: the production kernel ALREADY does exactly that. It is a single
    # kernel that overrides arena tokens with a computed tile
    # (`if tl.max(use_hp) > 0: hpv = tl.load(HP ...)`). So "fold it" is not an
    # unexplored direction, it is the very arm gf is measured against.
    #
    # The two are the two ends of one trade-off:
    #   production  folded, 1 kernel  -- ~2% of blocks take the arena branch,
    #                                    but registers are reserved in ALL of
    #                                    them
    #   gf          split, 2 kernels  -- no full-width tile in the fast path,
    #                                    but a fixed second-pass cost per step
    # Their crossovers are opposite, which is precisely why a context-gated
    # default is the right answer given these two implementations, and why
    # neither can be tuned into dominating the other.
    #
    # Range-splitting the loop (main body over [P, tail_start) with no HP code
    # at all) is NOT a safe third option: a token that has fallen out of the
    # recent window can still legitimately own its ring row when the ring has
    # not yet wrapped past it -- seq 700, P 64, tail 188, a token at 150 written
    # when seq was 600. Only 100 tokens have passed, the 512-row ring has not
    # wrapped, so the owner check still matches and a range-based skip would
    # drop a live token. That is why the tag check exists in every block.
    #
    # Removing the arena is not an option either: it is what keeps the newest
    # tokens in BF16. The zero-window run above is a diagnostic, not a config.
    #
    # REVERTED to the original constants 2026-08-26. The +10% is real as a
    # measurement, but it is UNVERIFIED for correctness and cannot be verified
    # with the gate as it stands: bench_kernel's build() hands the reference
    # path torch.empty logits/lse, so the reference can read uninitialized
    # slots. That shows up as nondeterministic catastrophic mismatches --
    # bs=1 seq=100 MATCHed at 6.5e-03 in one run and MISMATCHed at 9.3e-01 in
    # the next, on identical code, with several rel = 1.000e+00.
    #
    # So the honest state is: the gate's verdicts are not trustworthy right now,
    # which means neither "the tuning is safe" nor "the tuning breaks it" is
    # established. Ship the constants that have been in production, keep the
    # knobs for measurement, and fix the fixture before touching the defaults.
    SGLANG_OSCAR_MLA_WINDOW_BLOCK_H = EnvInt(16)
    SGLANG_OSCAR_MLA_WINDOW_BLOCK_N = EnvInt(32)
    SGLANG_OSCAR_MLA_WINDOW_WARPS = EnvInt(4)
    SGLANG_OSCAR_MLA_WINDOW_STAGES = EnvInt(2)
    # TRI-STATE, and the default value below is NOT the effective default.
    # Unset means AUTO: the backend turns gf on when the server's context length
    # is >= 8192, decided once at init. Setting it to 0 or 1 pins it.
    #
    # `EnvBool(False)` is the value `.get()` returns when nothing is set; the
    # backend asks `.is_set()` first, so the False here is only the fallback for
    # any caller that does not. Do not "simplify" this to a plain default --
    # that silently removes the auto rule.
    #
    # Why auto, and why on the context length rather than the batch's actual
    # sequence lengths: measured end to end on GLM-5.2-FP8, gf/production decode
    # tok/s is 0.877-0.967 at ctx 1000, ~1.0 at 2000, and 1.29-1.72 from 16k up,
    # rising with concurrency. So gf is a clear win at long context and a real
    # loss at short, and a CUDA graph's capture key is the batch size, not the
    # sequence length -- one graph serves seq=100 and seq=32000 alike, so a
    # per-batch branch cannot execute at replay. Per-server is the only
    # adaptivity that survives capture. Full table in triton_backend.py.
    SGLANG_OSCAR_MLA_PACKED_GF = EnvBool(False)
    # Runs the production kernel alongside the group-factored one on the same
    # call and logs the deviation. Expensive; a debugging instrument, not a
    # serving option.
    SGLANG_OSCAR_MLA_PACKED_GF_CHECK = EnvBool(False)
    # Shadow the packed latent with the BF16 fake-quant result and assert every
    # materialized read matches it. Doubles latent memory; for smoke runs only.
    SGLANG_OSCAR_MLA_PACKED_SELFCHECK = EnvBool(False)
    # How many reads the self-check verifies before switching itself off. Each
    # one re-materializes the row set and syncs the stream.
    SGLANG_OSCAR_MLA_PACKED_SELFCHECK_BUDGET = EnvInt(600)
    # Number of BF16 window rows per request in the packed pool's HP arena.
    # 0 = sink + recent (the two windows), which is the only correct value;
    # exposed so the arena can be sized down for probes.
    SGLANG_OSCAR_MLA_PACKED_HP_REQS = EnvInt(0)
    # Packed MLA decode kernel tiling. 16/8 rather than the BF16 kernel's 32/4
    # because the dequant keeps the code byte and the group scale/zero live on
    # top of the output tile.
    SGLANG_OSCAR_MLA_PACKED_BLOCK_N = EnvInt(16)
    # The group-factored kernel's own tile width. It used to read the knob
    # above, which meant it shipped at 16 -- the computed tile's optimum and
    # this kernel's PESSIMUM. Measured at warps=4 BLOCK_H=16, three shapes,
    # BLOCK_N the only variable: 32 is 1.61-1.68x faster than 16 everywhere,
    # with 64 in between, so the ordering is monotone rather than a single
    # lucky point. Separate knob because one value cannot serve both kernels.
    SGLANG_OSCAR_MLA_PACKED_GF_BLOCK_N = EnvInt(32)
    # 4, measured, not 8. Swept on the production computed-tile kernel at three
    # batch sizes, seq 20000, BLOCK_N=16 stages=1:
    #   bs=8   0.529 ms both        1.00x
    #   bs=16  0.917 vs 1.161       1.27x
    #   bs=32  1.811 vs 2.307       1.27x
    # Never worse, 1.27x better wherever it differs. The 8 was chosen to avoid
    # register spills and never benchmarked, so this is the first time the knob
    # has had a measurement behind it -- and it applies to the kernel serving
    # today, independent of the group-factored path.
    SGLANG_OSCAR_MLA_PACKED_WARPS = EnvInt(4)
    SGLANG_OSCAR_MLA_PACKED_STAGES = EnvInt(1)
    # Expand the per-group scale/zero in registers instead of addressing them as
    # a [BLOCK_N, D] tile through gid. Bit-identical -- the pack layout is
    # d = g * GS + j, so the broadcast reproduces gid element for element -- and
    # it removes 128x-redundant address computation, which is the cost that
    # survived the bandwidth refutation.
    #
    # A/B'd on hardware now, and it is SLOWER, so it stays off:
    #
    #     shape              bcast=0   bcast=1
    #     bs16 seq20000      0.916 ms  0.990 ms   -8%
    #     bs32 seq20000      1.806 ms  1.936 ms   -7%
    #
    # Correctness was fine (0 mismatches), so this is a pure speed refutation:
    # removing redundant address arithmetic does not pay when it costs the
    # registers that held it. Do not re-enable without a new measurement.
    SGLANG_OSCAR_MLA_PACKED_PARAM_BCAST = EnvBool(False)
    # Read the packed codes twice, once per dot layout, instead of transposing
    # one tile. The transpose is a shared-memory layout conversion; the second
    # dequant is free (measured: removing the dequant entirely is 1.00x). Off by
    # default until A/B'd on hardware.
    SGLANG_OSCAR_MLA_PACKED_DUAL_LOAD = EnvBool(False)
    # Audit the BF16 window arena during *decode*. The write-side self-check and
    # the teacher-forced NLL run both clear the packed read path, but teacher
    # forcing is a single prefill: the ring's owner tag, wrap and eviction are
    # never exercised. This counts, from pool state alone, how many of the slots
    # that should be BF16 a reader would actually accept.
    SGLANG_OSCAR_MLA_PACKED_AUDIT = EnvBool(False)
    SGLANG_OSCAR_MLA_PACKED_AUDIT_BUDGET = EnvInt(80)
    # Sample every Nth decode step: the tag only gets interesting after the ring
    # has wrapped, which takes thousands of steps.
    SGLANG_OSCAR_MLA_PACKED_AUDIT_STRIDE = EnvInt(200)
    # Audit every batch element, not just element 0. The failure being hunted is
    # an arena row reclaimed by whichever request now holds the same req index,
    # and that victim is not necessarily element 0.
    SGLANG_OSCAR_MLA_PACKED_AUDIT_ALL = EnvBool(False)
    # Opt Kimi-K3 into the honest MLA declaration (latent c_kv storage) instead
    # of the expanded per-head K/V 192/128 its published score was measured on.
    # Off by default until the latent path has its own scored K3 run: it also
    # needs per-layer 512x512 latent rotations, which do not exist for K3 yet.
    SGLANG_OSCAR_K3_MLA_LATENT = EnvBool(False)
    # OSCAR-for-latent high-precision subspace: dir of layer_<i>.pt files, each
    # [k, kv_lora_rank] orthonormal rows = the top-k most sensitivity-weighted
    # latent directions (from the kv_b_proj Hessian). Their projection is kept in
    # BF16; only the residual is rotated + INT2-quantized. Beats plain Hadamard.
    SGLANG_OSCAR_MLA_KV_HP_SUBSPACE_PATH = EnvStr("")
    SGLANG_OSCAR_K_CLIP_RATIO = EnvFloat(0.0)
    SGLANG_OSCAR_V_CLIP_RATIO = EnvFloat(0.0)
    SGLANG_OSCAR_ABSORB_V_ROTATION = EnvBool(False)
    # Fuse oscar K-rotation (rows @ R_k) into the prefill clip+quantize+pack
    # kernel. Eliminates the separate bf16 GEMM staging and the intermediate
    # rotated-K tensor for the quant pack. Requires oscar mode, V-rotation
    # absorbed (so V skips rotation), per-row scale (single-scale int2), and
    # at least one of K/V clip ratios > 0. Off by default; safe to leave off.
    SGLANG_OSCAR_FUSED_ROTATE_CLIP_QUANT = EnvBool(False)
    # Use Lloyd-Max MSE-optimal buckets for INT2 KV quantization instead of
    # the default uniform min-max. Applies only to single-scale pretransformed
    # clip kernels (num_groups == 1). Requires oscar rotation + clip enabled.
    SGLANG_LLOYD_MAX = EnvBool(False)
    SGLANG_MIXED_KV_HP_MAX_SPLITS = EnvInt(8)
    HADAMARD_ORDER = EnvInt(16)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    # Extra slots in req_to_token_pool for decode workers (only effective when
    # max_num_reqs > 32). Increases pool capacity so more KV cache transfers
    # can overlap with decode execution without raising max_running_requests.
    SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS = EnvInt(0)

    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB = EnvInt(64)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("tcp")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)

    # MPS (Apple Silicon)
    SGLANG_USE_MLX = EnvBool(False)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    SGLANG_NPU_FUSED_MOE_MODE = EnvInt(1)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2 = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # TODO(mmangkad): Remove this once the FlashInfer unified allreduce-fusion
    # transport issue on GB200/GB300 platforms is fixed and verified resolved.
    SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT = EnvBool(None)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)

    # DeepSeek MHA Optimization
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False)
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)

    # NIXL-EP
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)

    # NSA Backend
    SGLANG_NSA_FUSE_TOPK = EnvBool(True)
    SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA = EnvBool(True)
    SGLANG_USE_FUSED_METADATA_COPY = EnvBool(True)
    SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # Flash Attention
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)

    # vLLM dependencies (TODO: they have been deprecated, we can remove them safely)
    USE_VLLM_CUTLASS_W8A8_FP8_KERNEL = EnvBool(False)

    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_SPEC_V2 = EnvBool(False)
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)
    SGLANG_SPEC_NAN_DETECTION = EnvBool(False)
    SGLANG_SPEC_OOB_DETECTION = EnvBool(False)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)


    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(4 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # HTTP Server
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)

    # HTTP/2 Server
    SGLANG_GRANIAN_PARENT_PID = EnvInt(None)

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # Encoder gRPC
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(False)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer
    SGLANG_PATCH_TOKENIZER = EnvBool(False)  # TODO enable by default

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # fmt: on

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)

    # Elastic EP Backup Port
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # Sglang Cache Dir
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))


envs = Envs()
EnvField._allow_set_name = False


def _print_deprecated_env(new_name: str, old_name: str):
    if old_name in os.environ:
        warnings.warn(
            f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
        )
        os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_LOG_GC", "SGLANG_GC_LOG")
    _print_deprecated_env(
        "SGLANG_MOE_NVFP4_DISPATCH", "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError(f"assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
