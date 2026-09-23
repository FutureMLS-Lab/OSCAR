#!/usr/bin/env bash
# Generic GPQA eval driver for INT2 KV cache + OSCAR rotation.
#
# Required env:
#   MODEL          HuggingFace model id (e.g. Qwen/Qwen3-8B)
#   ROT_DIR        Folder containing {k,v}_rotation_qqt_r_h_pbr.pt
#   RUN_DIR        Output dir (logs + eval results)
#
# Optional env:
#   TP_SIZE        Tensor-parallel size for the eval server (default 4)
#   GPUS           CUDA_VISIBLE_DEVICES list (default 0,1,2,3)
#   PORT           HTTP port (default 31057)
#   DIST_PORT      Dist-init port (default 41057)
#   MEM_FRAC       --mem-fraction-static (default 0.8)
#   MAX_RUNNING    max-running-requests (default 64)
#   CUDA_GRAPH_MAX_BS (default 32)
#   GROUP_SIZE     int2 quant group size (default 128 — validated)
#   MAX_NEW_TOKENS (default 32768)
#   NUM_WORKERS    simple-evals client workers (default 32)
#   N_REPEATS      (default 1)
#   PRE_ROPE_FA3   set to 1 to force prefill fa3 + decode triton (default 1)

set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${MODEL:?MODEL is required}"
: "${ROT_DIR:?ROT_DIR is required}"
: "${RUN_DIR:?RUN_DIR is required}"

SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${REPO_ROOT}/sglang-research}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PORT="${PORT:-31057}"
DIST_PORT="${DIST_PORT:-41057}"
# Every other script in rotation/ spells this MEM_FRACTION_STATIC, so accept
# that name too -- a caller that used it got the 0.8 default with no warning.
MEM_FRAC="${MEM_FRAC:-${MEM_FRACTION_STATIC:-0.8}}"
MAX_RUNNING="${MAX_RUNNING:-64}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-32}"
# DISABLE_CUDA_GRAPH=1 serves without graph capture. The eager path is far
# slower, so this is a diagnostic knob, not a default: when a model garbles,
# it separates "the model is wrong" from "the capture is wrong", which has
# been the answer more than once in this codebase.
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--disable-cuda-graph)
else
    CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi
# V-rotation absorption folds R_v into o_proj. It assumes one rotation per
# layer, so it is invalid for per-head (format_version 2) checkpoints -- with
# per-head rotations and absorption on, Qwen3-30B-A3B scores 34.3 on GPQA
# against 58.6 with it off, i.e. worse than a shared rotation. Default off.
ABSORB_V="${ABSORB_V:-0}"
# MLA models (shared latent c_kv) rotate the latent, not per-head K/V, and read a
# different variable. Set MLA_ROT_PATH to the directory of per-layer layer_*.pt;
# the K/V paths above are then unused. Leaving it empty keeps the MHA path.
MLA_ROT_PATH="${MLA_ROT_PATH:-}"
# MLA models quantize the shared latent through the rotation path and keep the
# KV cache itself in bf16; passing --kv-cache-dtype int2 makes sglang abort with
# "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype".
GROUP_SIZE="${GROUP_SIZE:-128}"

# Whether this model is on the MLA (shared-latent) path has to be decided BEFORE
# the bf16 control clears the rotation path, because the two need different
# baselines: an MLA bf16 control still has to pin --kv-cache-dtype bfloat16
# (sglang would otherwise hand a DSA model fp8_e4m3 on SM100+ and the "bf16"
# baseline would silently be an fp8 one), while an MHA control wants plain auto.
IS_MLA=0; [[ -n "${MLA_ROT_PATH}" ]] && IS_MLA=1

if [[ "${KV_MODE:-int2}" == "bf16" ]]; then
    # The BF16 control for the speed comparison. It has to come from THIS script
    # rather than a second harness: the point of the measurement is the KV path,
    # so every other flag -- attention backends, radix cache, cuda-graph bs,
    # mem fraction, model, TP -- must be the ones the INT2 run used. A separate
    # bf16 harness would differ in several of them at once and the ratio would
    # not mean what it claims.
    if [[ "${IS_MLA}" == "1" ]]; then
        KV_DTYPE_ARGS=(--kv-cache-dtype "${MLA_KV_CACHE_DTYPE:-bfloat16}")
    else
        KV_DTYPE_ARGS=(--kv-cache-dtype auto)
    fi
    GROUP_SIZE_ARGS=()
    # Turn the latent quantizer off too. Leaving MLA_ROT_PATH set would keep
    # packed 2-bit latents on and the "bf16 baseline" would be measuring OSCAR.
    MLA_ROT_PATH=""; MLA_PACKED=0
    # For Kimi-K3 this arm is worth running precisely because it FAILS. With
    # the two cleared above, no OSCAR pool is built at all -- the run log shows
    # MLAPacked=0 and a stock latent cache at 1152 B/token/layer, exactly 4x
    # fewer tokens than the packed arm -- so this is plain sglang MLA, and it
    # scores 18.75 on GPQA-48 against 83.33 for the 2-bit arm and 95.83 for
    # stock upstream. The defect is in this fork's K3 MLA forward, and the
    # quantiser somehow masks it. Do not silence this arm: it is the only
    # configuration that shows the bug.
elif [[ -n "${MLA_ROT_PATH}" ]]; then
    # Arrays, not strings: an empty string still expands to one empty argv
    # entry, which sglang's argparse rejects as an unexpected positional.
    #
    # The MLA pool fake-quantizes into a normal float KV cache, so that cache's
    # dtype is the dtype its rotations and dequantized latents live in -- and it
    # otherwise follows the GPU generation, not the recipe: sglang gives a
    # DeepSeek-DSA model fp8_e4m3 on SM100+ and bfloat16 on Hopper and below.
    # The same eval then measures a different method per cluster (GLM-5.2-FP8
    # scored 68.7 on H100 and 5.6 on B200). Pin it; set MLA_KV_CACHE_DTYPE to
    # fp8_e4m3 deliberately if that is what you mean to measure.
    KV_DTYPE_ARGS=(--kv-cache-dtype "${MLA_KV_CACHE_DTYPE:-bfloat16}")
    # --kv-cache-quant-group-size is only accepted alongside int2, and the MLA
    # pool takes its group size from SGLANG_OSCAR_MLA_KV_GROUP_SIZE instead.
    GROUP_SIZE_ARGS=()
else
    KV_DTYPE_ARGS=(--kv-cache-dtype int2)
    # Kimi-K3 leaves the group size unset on purpose: its K head dim is
    # 192 = 3 x 64, and no group size divides both 192 and 128 usefully. Passing
    # a group size anyway is not a tuning choice there, it is a wrong geometry.
    if [[ "${NO_GROUP_SIZE:-0}" == "1" ]]; then
        GROUP_SIZE_ARGS=()
    else
        GROUP_SIZE_ARGS=(--kv-cache-quant-group-size "${GROUP_SIZE}")
    fi
fi
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
# Prefix caching. This eval used to pin --disable-radix-cache because mixed-KV
# tiering and the radix cache corrupted each other. Two causes, both fixed:
#   * CUDA-graph padded decode writes landed in HP-prefix slot 0, which was
#     allocatable -- and with the cache on that page is normally part of the
#     *shared* prefix node, so one padded replay corrupted the prefix every
#     live request reads. HP-prefix page 0 is now a reserved padding sink.
#     (This was the big one: 27.78 vs 57.07 on Qwen3-30B-A3B GPQA-198.)
#   * A cached prefix could reach into the borrower's BF16 HP-recent window,
#     serving it at 2 bits; RadixCache._mixed_kv_tier_cap now bounds every
#     match, internal ones included.
# Cache on by default: 60.61 with 198/198 answered and a 17.1% token hit rate.
# Set DISABLE_RADIX=1 for the old behavior (and to A/B it).
if [[ "${DISABLE_RADIX:-0}" == "1" ]]; then
    RADIX_ARGS=(--disable-radix-cache)
else
    RADIX_ARGS=()
fi
# Attention backends. FA3's int2 prefill asserts on sliding-window layers, so a
# model with local attention (Gemma-4 has 40 such layers) must set both of these
# to triton -- the int2 prefill path reads the global one, so overriding only
# PREFILL_BACKEND is not enough.
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
PREFILL_BACKEND="${PREFILL_BACKEND:-fa3}"
# Multi-node. The 400B-class models (MiniMax-M3, GLM-5.2-FP8) do not fit on one
# node, so without these the example cannot run them at all. Set NNODES>1 plus
# NODE_RANK and DIST_ADDR on each node; only rank 0 drives the eval.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
DIST_ADDR="${DIST_ADDR:-}"
if [[ "${NNODES}" -gt 1 ]]; then
    [[ -n "${DIST_ADDR}" ]] || { echo "[eval-oscar] NNODES>1 needs DIST_ADDR=<head-ip>:<port>" >&2; exit 1; }
    MULTINODE_ARGS=(--nnodes "${NNODES}" --node-rank "${NODE_RANK}" --dist-init-addr "${DIST_ADDR}")
    # rotation/run/kimi-k3.sh exported DIST_TIMEOUT and explained in its own
    # comment why it is needed -- "the two ranks reach the rendezvous minutes
    # apart, so --dist-timeout must exceed torch's 600 s default or the leader
    # gives up with 8/16 clients joined" -- but nothing here ever read it, so the
    # flag never reached the server and the documented failure stayed possible.
    [[ -n "${DIST_TIMEOUT:-}" ]] && MULTINODE_ARGS+=(--dist-timeout "${DIST_TIMEOUT}")
else
    # Only single-node gets the loopback rendezvous. argparse is last-wins,
    # so emitting this after MULTINODE_ARGS made loopback beat the real head
    # address and NNODES>1 could never rendezvous -- the documented
    # multi-node support for MiniMax-M3 / GLM-5.2 could not work at all.
    MULTINODE_ARGS=(--dist-init-addr "127.0.0.1:${DIST_PORT}")
fi
NUM_WORKERS="${NUM_WORKERS:-32}"
N_REPEATS="${N_REPEATS:-1}"
NAME="${NAME:-gpqa_oscar}"

# Respect an environment the caller already activated. Gemma-4 needs
# transformers >= 5.5 for Gemma4TextConfig and runs from its own venv; forcing
# the default conda env here silently reverted it, and sglang then reported the
# model as "not a registered model" because gemma4_unified failed to import.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[eval-oscar] using pre-activated venv ${VIRTUAL_ENV}"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
else
    CONDA_BASE="${CONDA_BASE:-${HOME}/miniconda3}"
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-oscar}"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi
# Prepend per-rank Triton cache redirector so TP workers don't race on shared
# launcher .so / metadata files in TRITON_CACHE_DIR.
export PYTHONPATH="${REPO_ROOT}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}"
# Rank-suffixed under multi-node: every rank used to open ${RUN_DIR}/server.log
# and truncate it, so the two ranks clobbered each other and a two-node failure
# left one interleaved file that could not say which rank died first. Rank 0
# keeps the plain name so single-node runs and every existing reader are
# unaffected.
LOG_SERVER="${RUN_DIR}/server.log"
[[ "${NNODES:-1}" -gt 1 && "${NODE_RANK:-0}" != "0" ]] && LOG_SERVER="${RUN_DIR}/server.rank${NODE_RANK}.log"
LOG_RUNNER="${RUN_DIR}/runner.log"   # streaming stdout from the eval runner
# run_simple_eval.py writes the canonical pretty-table eval.log to ${RUN_DIR}/.
: > "${LOG_SERVER}"

# Per-run Triton cache to avoid races when multiple eval servers compile the
# same kernel name into the shared default cache (~/.triton/cache).
# OSCAR_TRITON_PER_RANK_BASE is read by sitecustomize.py to route each TP
# rank into its own subdir (rank0/, rank1/, ...) — breaks intra-job races.
export OSCAR_TRITON_PER_RANK_BASE="${OSCAR_TRITON_PER_RANK_BASE:-${RUN_DIR}/triton_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OSCAR_TRITON_PER_RANK_BASE}/main}"
mkdir -p "${OSCAR_TRITON_PER_RANK_BASE}" "${TRITON_CACHE_DIR}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

SERVER_ARGS=(
    --model-path "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --attention-backend "${ATTN_BACKEND:-fa3}"
    --prefill-attention-backend "${PREFILL_BACKEND:-fa3}"
    # Overridable. INT2 needs the triton decode backend -- that is where the
    # OSCAR kernels live -- but a BF16 control has no such requirement, and
    # hardcoding it silently overrode the per-model default sglang picks for
    # itself (for Kimi-K3 on SM100/SM103 that default is trtllm_mla). A
    # control arm that cannot be put on the stock path cannot be compared to
    # stock sglang, which is exactly the comparison K3 needed.
    --decode-attention-backend "${DECODE_BACKEND:-triton}"
    # Vision tower. On Blackwell vision.py defaults to "fa4", whose kernels come
    # from flash_attn.cute -- and that module imports against a cutlass-dsl whose
    # API moved (cutlass.cute.core has no attribute 'ThrMma'). sglang reports it
    # as "Vendored FlashAttention CUTE is not available", which reads like a
    # missing file rather than a version skew. Qwen3.5-4B routes through
    # qwen3_vl.py, so it dies at the first forward even for a text-only eval.
    #
    # This was the THIRD flag that existed only in rotation/verify/. The first
    # two (--disable-flashinfer-autotune, --mamba-scheduler-strategy) were added
    # above after three models failed GPQA having passed the smoke harness. Any
    # flag added to verify/mha.sh from here on belongs in this list too --
    # otherwise the harness that says a model works and the harness that scores
    # it are testing two different servers.
    --mm-attention-backend "${MM_ATTENTION_BACKEND:-triton_attn}"
    "${MULTINODE_ARGS[@]}"
    "${KV_DTYPE_ARGS[@]}"
    "${GROUP_SIZE_ARGS[@]}"
    --mem-fraction-static "${MEM_FRAC}"
    --max-running-requests "${MAX_RUNNING}"
    "${RADIX_ARGS[@]}"
    --enable-cache-report
    "${CUDA_GRAPH_ARGS[@]}"
    --host 127.0.0.1
    --port "${PORT}"
    --trust-remote-code
)
# FlashInfer autotune is OFF by default here.
#
# _flashinfer_autotune() issues a _dummy_run at batch_size = req_to_token_pool.size.
# INT2 pools are small (hybrid Qwen3.5-4B: 2582 tokens, because the Mamba state
# takes 41 GB), so that dummy batch indexes past the end of the pool and faults.
# The fault is ASYNCHRONOUS, so it surfaces later at whatever kernel the host
# next synchronizes on -- here `load_binary` of a decode stage-1 kernel, which
# reads as a Triton/ptxas problem and cost eleven wrong hypotheses. Autotune buys
# nothing for this eval: the decode backend is triton, not flashinfer.
#
# This lived only in rotation/verify/ before, which is why every model passed the
# smoke harness and three then failed GPQA. Two launch paths with different flags
# is the bug; one place to set them is the fix.
if [[ "${FLASHINFER_AUTOTUNE:-0}" != "1" ]]; then
    SERVER_ARGS+=(--disable-flashinfer-autotune)
fi
# Hybrid (linear-attention) models select MambaRadixCache, whose page_size==1
# assertion the INT2 page-8 layout violates; the assertion is guarded by
# `if not self.enable_mamba_extra_buffer`, so extra_buffer keeps the prefix cache
# on. The run scripts exported this variable and nothing read it.
if [[ -n "${MAMBA_SCHEDULER_STRATEGY:-}" ]]; then
    SERVER_ARGS+=(--mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}")
fi
if [[ -n "${REASONING_PARSER:-}" ]]; then
    SERVER_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    SERVER_ARGS+=(${EXTRA_SERVER_ARGS})
fi

echo "[eval-oscar] model=${MODEL} tp=${TP_SIZE} gpus=${GPUS} rot=${ROT_DIR} out=${RUN_DIR}"
# SGLANG_LLOYD_MAX is PASSED ONLY IF THE CALLER SET IT.
#
# It used to be `SGLANG_LLOYD_MAX="${LLOYD_MAX:-0}"` in the prefix below, which
# always put the variable in the environment -- so `is_set()` was always True
# and the pool's own default never ran. That matters now that the packed MLA
# pools default Lloyd-Max ON at 2 bits: forcing the variable silently pinned
# every eval to the uniform codebook, which on GLM-5.2 is 76.67% against
# 83.33%. A harness that hardcodes a default cannot test that default, and it
# reports the shipped configuration's number for a configuration nobody ships.
#
# Same shape of bug as gf_speed setting SGLANG_OSCAR_MLA_PACKED_GF in both
# arms: whenever a harness supplies a value "for reproducibility", the branch
# that runs when nobody supplies one stops being exercised.
OPT_ENV=()
if [ -n "${LLOYD_MAX:-}" ]; then
    OPT_ENV+=("SGLANG_LLOYD_MAX=${LLOYD_MAX}")
    echo "[eval-oscar] SGLANG_LLOYD_MAX pinned to ${LLOYD_MAX} by the caller"
else
    echo "[eval-oscar] SGLANG_LLOYD_MAX left UNSET -- the pool's own default applies"
fi
SGLANG_ENABLE_MIXED_KV_WINDOWS="$([[ "${KV_MODE:-int2}" == "bf16" ]] && echo 0 || echo 1)" \
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_COQUANT_ROTATION_MODE=coquant \
SGLANG_OSCAR_ABSORB_V_ROTATION="${ABSORB_V:-0}" \
SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
SGLANG_MIXED_KV_PREFIX_TOKENS=${SGLANG_MIXED_KV_PREFIX_TOKENS:-64} \
SGLANG_MIXED_KV_RECENT_TOKENS=${SGLANG_MIXED_KV_RECENT_TOKENS:-256} \
SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${HP_PREFIX_POOL_TOKENS:-0}" \
SGLANG_MIXED_KV_AUDIT="${MIXED_KV_AUDIT:-0}" \
SGLANG_MIXED_KV_AUDIT_EVERY="${MIXED_KV_AUDIT_EVERY:-25}" \
SGLANG_OSCAR_MLA_KV_ROTATION_PATH="${MLA_ROT_PATH:-}" \
SGLANG_OSCAR_MLA_KV_GROUP_SIZE="${MLA_GROUP_SIZE:-128}" \
SGLANG_OSCAR_MLA_KV_PACKED="${MLA_PACKED:-0}" \
SGLANG_OSCAR_MLA_PACKED_SELFCHECK="${MLA_PACKED_SELFCHECK:-0}" \
SGLANG_OSCAR_K_ROTATION_PATH="${ROT_DIR}/${K_ROT_FILENAME:-k_rotation_qqt_r_h_pbr.pt}" \
SGLANG_OSCAR_V_ROTATION_PATH="${ROT_DIR}/${V_ROT_FILENAME:-v_rotation_sst_r_h_pbr.pt}" \
SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP:-0.96}" \
SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP:-0.92}" \
CUDA_VISIBLE_DEVICES="${GPUS}" \
env "${OPT_ENV[@]}" \
python -m sglang.launch_server "${SERVER_ARGS[@]}" >> "${LOG_SERVER}" 2>&1 &
SERVER_PID=$!


# Only rank 0 has the HTTP server and drives the eval; the other ranks just serve
# their shard. Without this they wait on a health endpoint that never comes up
# and then tear the group down, which surfaces on rank 0 as
# "DistNetworkError: Failed to recv, got 0 bytes".
if [[ "${NODE_RANK}" != "0" ]]; then
    echo "[eval-oscar] rank ${NODE_RANK}: serving only, waiting for the group"
    wait "${SERVER_PID}"
    exit 0
fi
# 400B-class models at TP=16 spend ~1220 s just loading weights, so the old
# fixed 240x5s = 20 min ceiling killed them mid-load.
HEALTH_WAIT_STEPS="${HEALTH_WAIT_STEPS:-240}"
for _ in $(seq 1 "${HEALTH_WAIT_STEPS}"); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[eval-oscar] server ready"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval-oscar] server died"
        tail -100 "${LOG_SERVER}" || true
        exit 1
    fi
    sleep 5
done

if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[eval-oscar] server not ready after 20 min"
    tail -100 "${LOG_SERVER}" || true
    exit 1
fi

# BENCH_PREFILL swaps the client, not the server. Reusing the launch path above
# is the entire point: the 64K number and the GPQA number then come from one
# configuration, so a speed claim cannot quietly describe a setup that was never
# scored.
if [[ "${BENCH_PREFILL:-0}" == "1" ]]; then
    echo "[eval-oscar] decode-at-context benchmark: ${BENCH_PREFILL_TOKENS:-65536} ctx tokens, mode=${KV_MODE:-int2}"
    python "${REPO_ROOT}/rotation/_eval_runner/bench_decode64k.py" \
        --port "${PORT}" \
        --prefill-tokens "${BENCH_PREFILL_TOKENS:-65536}" \
        --max-new-tokens "${BENCH_NEW_TOKENS:-512}" \
        --reps "${BENCH_REPS:-3}" \
        --label "${NAME}:${KV_MODE:-int2}" \
        --out "${RUN_DIR}/bench_${KV_MODE:-int2}.json" 2>&1 | tee -a "${LOG_RUNNER}"
    rc=${PIPESTATUS[0]}
    exit "${rc}"
fi

echo "[eval-oscar] launching eval via simple_evals (vendored at third_party/simple_evals)"
# A fresh clone leaves third_party/simple_evals empty (it is a submodule) and
# every grader imports it, so the eval dies with
#   ImportError: cannot import name 'common' from 'simple_evals'
# only after the model is loaded. Initialise it up front and fail loudly.
SE="${REPO_ROOT}/third_party/simple_evals"
if [[ ! -f "${SE}/common.py" ]]; then
    echo "[eval-oscar] third_party/simple_evals is empty; initialising submodule"
    git -C "${REPO_ROOT}" submodule update --init --recursive third_party/simple_evals || true
    find "${SE}" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
fi
if [[ ! -f "${SE}/common.py" ]]; then
    echo "[eval-oscar] cannot find ${SE}/common.py -- run: git submodule update --init --recursive" >&2
    exit 1
fi

RUNNER="${REPO_ROOT}/rotation/_eval_runner/run_simple_eval.py"
# TASK is overridable so the same launch path can run a short-answer benchmark
# against the same server. GPQA's long thinking generations conflate two failure
# modes -- a model that answers wrongly and a model that never stops -- and
# math500/humaneval separate them.
python "${RUNNER}" \
    --task "${TASK:-gpqa}" \
    --model "${MODEL}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --max-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --top-p "${TOP_P:-0.95}" \
    --top-k "${TOP_K:-40}" \
    --presence-penalty "${PRESENCE_PENALTY:-0}" \
    ${THINKING_EFFORT:+--thinking-effort "${THINKING_EFFORT}"} \
    --n-repeats "${N_REPEATS}" \
    --num-threads "${NUM_WORKERS:-32}" \
    ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_RUNNER}"
echo "[eval-oscar] done. score:"
grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" | tail -10 || true
