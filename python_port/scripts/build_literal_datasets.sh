#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STRICT_FLAG=""
if [[ "${STRICT:-0}" == "1" ]]; then
  STRICT_FLAG="--strict"
fi

echo "Generating literal JSONL files from story splits..."
for SPLIT in train val test; do
  INPUT="data/${SPLIT}.json"
  OUTPUT="data/literals_${SPLIT}.jsonl"
  if [[ ! -f "$INPUT" ]]; then
    echo "Warning: ${INPUT} not found, skipping."
    continue
  fi
  echo "  -> ${OUTPUT}"
  uv run python -m nls_python.model_generation.annotated_dataset \
    --inputs "$INPUT" \
    --output "$OUTPUT" \
    $STRICT_FLAG
done
