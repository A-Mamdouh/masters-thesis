#!/usr/bin/env python3
"""Plot literal seq2seq validation/test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Metrics file must contain a JSON object.")
    return data


def plot(metrics: Dict[str, Dict[str, float]], output: Path) -> None:
    splits = sorted(metrics.keys())
    exact = [metrics[split].get("exact_match", 0.0) for split in splits]
    topk = [metrics[split].get("topk_accuracy", 0.0) for split in splits]
    import numpy as np

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, exact, width, label="Exact match")
    ax.bar(x + width / 2, topk, width, label="Top-k accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Literal seq2seq performance")
    ax.legend()
    for idx, value in enumerate(exact):
        ax.text(x[idx] - width / 2, value + 0.02, f"{value:.2f}", ha="center")
    for idx, value in enumerate(topk):
        ax.text(x[idx] + width / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot literal seq2seq metrics.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to literal_metrics.json.")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path.")
    args = parser.parse_args()
    metrics = load_metrics(args.metrics)
    if not metrics:
        raise SystemExit("Metrics file is empty.")
    plot(metrics, args.output)
    print(f"Wrote literal metrics plot to {args.output}")


if __name__ == "__main__":
    main()
