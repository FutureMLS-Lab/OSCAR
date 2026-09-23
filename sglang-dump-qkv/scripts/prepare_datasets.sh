#!/bin/bash
# prepare_datasets.sh — Download and convert datasets required for eval_kv_rotation.sh
#
# Usage: bash prepare_datasets.sh <python_executable> <script_dir>

set -eo pipefail

PYTHON="${1:-python3}"
SCRIPT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# =============================================================================
# aime25 (math-ai/aime25 -> togethercomputer/math_problems schema)
# =============================================================================
if [ ! -d "$SCRIPT_DIR/datasets/aime25" ]; then
    echo "Preparing aime25 dataset (math-ai/aime25)..."
    mkdir -p "$SCRIPT_DIR/datasets"
    "$PYTHON" - "$SCRIPT_DIR/datasets/aime25" <<'PYEOF'
import sys
from datasets import load_dataset

out_path = sys.argv[1]
ds = load_dataset("math-ai/aime25", split="test")
converted = ds.map(
    lambda x: {
        "prompt": x["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "ground_truth": str(x["answer"]),
        "data_source": "aime25",
        "extra_info": {},
    },
    remove_columns=ds.column_names,
)
converted.save_to_disk(out_path)
print(f"✓ aime25 saved ({len(converted)} examples) -> {out_path}")
PYEOF
else
    echo "✓ aime25 already present"
fi
