#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATA_ROOT="${DATA_ROOT:-experiments/assistant_v03_pilot_seed31}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/lora_pilot_seed31}"

for condition in neutral self other shuffled_self spp; do
  echo "===== training ${condition} ====="
  python scripts/train_condition_lora.py \
    --condition "${condition}" \
    --data-root "${DATA_ROOT}" \
    --base-model "${MODEL}" \
    --output-root "${OUTPUT_ROOT}"
done
