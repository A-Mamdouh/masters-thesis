#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICIES_DEFAULT="CustomJava"
DATASETS_DEFAULT="data/train.json"

I_FS="${POLICIES:-$POLICIES_DEFAULT}"
D_FS="${DATASETS:-$DATASETS_DEFAULT}"

read -r -a POLICY_ARGS <<< "$I_FS"
read -r -a DATASET_ARGS <<< "$D_FS"

OUTPUT_DIR="${POLICY_OUTPUT_DIR:-$ROOT_DIR/experiments/policies}"
POOL_SIZE="${POOL_SIZE:-1}"
TOPK="${TOPK:-5}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
TIMEOUT="${TIMEOUT:-600}"
NEURAL_DRAIN_SIZE="${NEURAL_DRAIN_SIZE:-1}"

SUMMARY="${OUTPUT_DIR}/summary.json"

uv run python -m nls_python.model_generation.policy_experiments \
  --pool-size "$POOL_SIZE" \
  --neural-drain-size "$NEURAL_DRAIN_SIZE" \
  --policies "${POLICY_ARGS[@]}" \
  --datasets "${DATASET_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --top-k "$TOPK" \
  --timeout-ms "$TIMEOUT" \
  --summary "$SUMMARY"

uv run python scripts/plot_policy_metrics.py \
  --summary "$SUMMARY" \
  --output "${OUTPUT_DIR}/metrics.png"

echo "Policy experiments complete. Summary: $SUMMARY"
