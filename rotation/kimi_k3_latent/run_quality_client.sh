#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/shared/oscar-int3-rubin-simulate}"
SOURCE="${SOURCE:-${ROOT}/source/multimodel/CoQuant}"
BASE="${BASE:-${ROOT}/multimodel/kimi-k3}"
BASE_URL="${BASE_URL:-http://kimi-k3-latent-head:30000}"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
BLOCKS="${BLOCKS:-/shared/CoQuant/kimi3-official-2bit/wikitext2_test_8192.pt}"
TASK="${TASK:?TASK is required}"
MODE="${MODE:-calibration}"

verify_mode() {
  if [[ "${SKIP_CONTROL_MODE_CHECK:-0}" == "1" ]]; then
    return
  fi
  [[ "$(<"${BASE}/control/mode")" == "${MODE}" ]]
}

for _ in $(seq 1 720); do
  if curl -fsS "${BASE_URL}/model_info" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
curl -fsS "${BASE_URL}/model_info" >/dev/null

case "${TASK}" in
  calibration)
    mkdir -p "${BASE}/calibration"
    python3 - "${BASE_URL}" "${BLOCKS}" "${BASE}/calibration/request.json" <<'PY'
import json
import os
import sys
from pathlib import Path

import requests
import torch

base_url, blocks_path, output_path = sys.argv[1:]
block = torch.load(blocks_path, map_location="cpu", weights_only=True)["blocks"][0]
block = block[: int(os.environ.get("CALIBRATION_TOKENS", "4096"))]
payload = {
    "input_ids": [int(value) for value in block],
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
}
response = requests.post(
    base_url.rstrip("/") + "/generate", json=payload, timeout=7200
)
response.raise_for_status()
Path(output_path).write_text(json.dumps(response.json(), indent=2) + "\n")
print("KIMI_LATENT_CALIBRATION_REQUEST_OK", flush=True)
PY
    python3 - "${BASE}/calibration/latent" <<'PY'
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
layers = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
          51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92]
for _ in range(720):
    paths = [root / f"layer_{layer}" / "latent" / "0.pt" for layer in layers]
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        print("KIMI_LATENT_CALIBRATION_DUMPS_READY", flush=True)
        break
    time.sleep(5)
else:
    missing = [str(path) for path in paths if not path.is_file()]
    raise TimeoutError(f"Calibration dump copy timed out: {missing}")
PY
    ;;
  kl_teacher)
    mkdir -p "${BASE}/kl_8k/bf16"
    python3 "${SOURCE}/rotation/kimi_k3_latent/wikitext2_kl_tail.py" \
      --base-url "${BASE_URL}" \
      --model moonshotai/Kimi-K3-native \
      --blocks-path "${BLOCKS}" \
      --output "${BASE}/kl_8k/bf16/reference.json" \
      --top-k 50 --max-blocks 16 --positions-per-block 16 \
      --workers "${KL_WORKERS:-2}" --timeout 7200
    ;;
  ppl)
    [[ "${MODE}" == offline_lm_qp || "${MODE}" == oscar_offline_lm_qp ]]
    verify_mode
    mkdir -p "${BASE}/ppl/${MODE}"
    python3 "${SOURCE}/rotation/wikitext2_ppl.py" \
      --base-url "${BASE_URL}" \
      --model "Kimi-K3-${MODE}" \
      --tokenizer-path "${MODEL}" \
      --blocks-path "${BLOCKS}" \
      --block-size 8192 --score-tail-tokens 2048 \
      --max-blocks 16 --timeout 7200 \
      --output "${BASE}/ppl/${MODE}/ppl.json"
    ;;
  kl_student)
    [[ "${MODE}" == offline_lm_qp || "${MODE}" == oscar_offline_lm_qp ]]
    verify_mode
    for _ in $(seq 1 8640); do
      [[ -s "${BASE}/kl_8k/bf16/reference.json" ]] && break
      sleep 5
    done
    test -s "${BASE}/kl_8k/bf16/reference.json"
    mkdir -p "${BASE}/kl_8k/${MODE}"
    python3 "${SOURCE}/rotation/kimi_k3_latent/wikitext2_kl_tail.py" \
      --base-url "${BASE_URL}" \
      --model "Kimi-K3-${MODE}" \
      --blocks-path "${BLOCKS}" \
      --teacher-reference "${BASE}/kl_8k/bf16/reference.json" \
      --output "${BASE}/kl_8k/${MODE}/kl.json" \
      --top-k 50 --max-blocks 16 --positions-per-block 16 \
      --workers "${KL_WORKERS:-2}" --timeout 7200
    ;;
  gpqa)
    [[ "${MODE}" == offline_lm_qp || "${MODE}" == oscar_offline_lm_qp ]]
    verify_mode
    SEED="${SEED:?SEED is required for GPQA}"
    SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX is required for GPQA}"
    NUM_SHARDS="${NUM_SHARDS:-4}"
    RUN_DIR="${BASE}/gpqa/${MODE}/seed${SEED}/shard${SHARD_INDEX}"
    mkdir -p "${RUN_DIR}"
    python3 "${SOURCE}/rotation/_eval_runner/run_simple_eval.py" \
      --task gpqa \
      --model "${MODEL}" \
      --base-url "${BASE_URL}/v1" \
      --max-tokens 32768 \
      --request-timeout 14400 \
      --temperature 1.0 --top-p 0.95 --top-k 40 \
      --seed "${SEED}" --n-repeats 1 \
      --num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX}" \
      --num-threads "${GPQA_WORKERS:-2}" \
      --output-dir "${RUN_DIR}"
    test -s "${RUN_DIR}/metrics.json"
    ;;
  *)
    echo "Unsupported TASK=${TASK}" >&2
    exit 2
    ;;
esac
