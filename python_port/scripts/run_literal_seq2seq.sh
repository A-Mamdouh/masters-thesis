#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN="${TRAIN:-data/literals_train.jsonl}"
VAL="${VAL:-data/literals_val.jsonl}"
TEST="${TEST:-data/literals_test.jsonl}"
MODEL_NAME="${MODEL_NAME:-facebook/bart-base}"
OUTPUT_DIR="${LITERAL_OUTPUT_DIR:-$ROOT_DIR/experiments/literal_seq2seq}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BEAM_SIZE="${BEAM_SIZE:-5}"
TOPK="${TOPK:-5}"
DEVICE="${DEVICE:-auto}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$DEVICE" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

uv run python -m nls_python.model_generation.literal_seq2seq \
  --train "$TRAIN" \
  --val "$VAL" \
  --test "$TEST" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --beam-size "$BEAM_SIZE" \
  --top-k "$TOPK" \
  --device "$DEVICE"

uv run python scripts/plot_literal_metrics.py \
  --metrics "${OUTPUT_DIR}/literal_metrics.json" \
  --output "${OUTPUT_DIR}/literal_metrics.png"

echo "Literal seq2seq training finished. Check ${OUTPUT_DIR} for checkpoints, metrics, and TensorBoard logs."
