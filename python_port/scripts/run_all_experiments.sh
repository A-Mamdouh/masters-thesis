#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[2/3] Running annotated test-set experiments"
uv run python -m nls_python.model_generation.annotated_story_experiments \
  --pool-size 1 \
  --timeout-ms 1200 \
  --input data/test.json \
  --base-policy CustomJava \
  --policies DFS BFS \
  --neural-checkpoint experiments/policies/CustomJava.pt \
  --neural-drain-size 1 \
  --output experiments/policies/annotated_test_results.txt
  

echo "[3/3] Evaluating neural policy vs. base policy"
uv run python -m nls_python.model_generation.story_experiments \
  --pool-size 1 \
  --timeout-ms 1200 \
  --base-policy CustomJava \
  --policies NeuralPolicy DFS BFS \
  --neural-checkpoint experiments/policies/CustomJava.pt \
  --neural-drain-size 1 \
  --output experiments/policies/custom_vs_neural.txt
  

echo "All experiments finished."
