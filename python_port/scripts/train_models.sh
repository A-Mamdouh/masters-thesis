#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/2] Running symbolic policy distillation"
./scripts/run_policy_experiments.sh

echo "[2/2] Training literal seq2seq model"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} DEVICE=${DEVICE:-auto} ./scripts/run_literal_seq2seq.sh

echo "Training complete"
