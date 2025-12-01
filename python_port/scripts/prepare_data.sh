#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Splitting story dataset"
uv run python splitter.py

echo "[2/3] Generating literal JSONL splits"
uv run ./scripts/build_literal_datasets.sh

echo "[3/3] Summary"
ls -1 data/train.json data/val.json data/test.json data/literals_train.jsonl data/literals_val.jsonl data/literals_test.jsonl
