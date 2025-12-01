#!/usr/bin/env python3
"""Generate summary plots for policy experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Summary JSON must be a list of objects.")
    return data


def plot_metrics(summary: List[Dict[str, Any]], output: Path) -> None:
    policies = [entry["policy"] for entry in summary]
    model_found = [entry["metrics"]["model_found_rate"] for entry in summary]
    topk = [entry["metrics"]["topk_accuracy"] for entry in summary]
    spearman = [entry["metrics"]["avg_spearman"] for entry in summary]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    metrics = [
        ("Model found rate", model_found),
        ("Top-k accuracy", topk),
        ("Average Spearman", spearman),
    ]
    for ax, (title, values) in zip(axes, metrics):
        ax.bar(policies, values, color="tab:blue")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Score")
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.02, f"{value:.2f}", ha="center")
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot policy experiment metrics.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json from run_policy_experiments.")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path.")
    args = parser.parse_args()
    data = load_summary(args.summary)
    if not data:
        raise SystemExit("Summary file is empty.")
    plot_metrics(data, args.output)
    print(f"Wrote policy metrics plot to {args.output}")


if __name__ == "__main__":
    main()
