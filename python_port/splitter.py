#!/usr/bin/env python3
"""Story dataset splitter.

Splits `data/data.json` into train/val/test sets without sentence-prefix leakage by
assigning whole stories to each split. The default strategy mirrors the domain
allocation we discussed earlier so every thematic bucket stays represented.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, TypeAlias

DEFAULT_DOMAIN_TARGETS: dict[str, dict[str, int]] = {
    "business": {"train": 7, "val": 1, "test": 1},
    "heist": {"train": 5, "val": 1, "test": 1},
    "investigation": {"train": 3, "val": 1, "test": 1},
    "emergency": {"train": 3, "val": 1, "test": 1},
    "celebration": {"train": 1, "val": 1, "test": 1},
    "unknown": {"train": 1, "val": 0, "test": 0},
}

Story: TypeAlias = dict[str, Any]
SplitDict: TypeAlias = dict[str, list[Story]]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Split story dataset into train/val/test JSON files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "data.json",
        help="Path to the full dataset JSON (default: data/data.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data",
        help="Directory where train/val/test JSON files will be written (default: data/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--domain-targets",
        type=Path,
        help=(
            "Optional JSON file mapping each domain to {\"train\": int, \"val\": int, \"test\": int}. "
            "If omitted, the built-in balanced targets are used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without writing any files.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[Story]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(f"Dataset not found: {path}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"Dataset must be a list of stories, got {type(data)!r}")
    return data


def load_domain_targets(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return DEFAULT_DOMAIN_TARGETS
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(f"Domain targets file not found: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("Domain targets JSON must be an object/dict.")
    targets: dict[str, dict[str, int]] = {}
    for domain, splits in payload.items():
        if not isinstance(splits, Mapping):
            raise SystemExit(f"Targets for domain '{domain}' must be an object.")
        try:
            train = int(splits["train"])
            val = int(splits["val"])
            test = int(splits["test"])
        except KeyError as exc:
            raise SystemExit(f"Domain '{domain}' missing key {exc!s}") from exc
        targets[domain] = {"train": train, "val": val, "test": test}
    return targets


def normalize_domain(story: Story) -> str:
    domain = story.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain
    return "unknown"


def split_stories(
    stories: Iterable[Story],
    *,
    domain_targets: Mapping[str, Mapping[str, int]],
    seed: int,
) -> SplitDict:
    buckets: MutableMapping[str, list[Story]] = defaultdict(list)
    for story in stories:
        buckets[normalize_domain(story)].append(story)

    rng = random.Random(seed)
    splits: SplitDict = {"train": [], "val": [], "test": []}

    for domain, targets in domain_targets.items():
        available = buckets.get(domain, [])
        required = sum(targets.values())
        if len(available) < required:
            raise SystemExit(
                f"Domain '{domain}' requires {required} stories but only {len(available)} present."
            )
        rng.shuffle(available)
        train_n = targets["train"]
        val_n = targets["val"]
        test_n = targets["test"]
        splits["train"].extend(available[:train_n])
        splits["val"].extend(available[train_n : train_n + val_n])
        splits["test"].extend(
            available[train_n + val_n : train_n + val_n + test_n]
        )
        buckets[domain] = available[train_n + val_n + test_n :]

    # Any extra domains (or leftovers beyond the requested counts) fall back to train.
    leftovers = sum(len(items) for domain, items in buckets.items() if items)
    if leftovers:
        sys.stderr.write(
            f"Warning: assigning {leftovers} leftover story/ies to the train split.\n"
        )
        for items in buckets.values():
            splits["train"].extend(items)

    return splits


def write_split(stories: list[Story], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(stories, fh, indent=4)
        fh.write("\n")


def summarize_split(splits: SplitDict) -> str:
    parts = []
    for split_name, subset in splits.items():
        parts.append(f"{split_name}: {len(subset)} stories")
    return ", ".join(parts)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.input)
    domain_targets = load_domain_targets(args.domain_targets)
    splits = split_stories(dataset, domain_targets=domain_targets, seed=args.seed)

    print(summarize_split(splits))
    if args.dry_run:
        for split_name, subset in splits.items():
            print(f"\n{split_name.upper()} ({len(subset)} stories)")
            for idx, story in enumerate(subset, start=1):
                synopsis = story.get("synopsis") or "(no synopsis provided)"
                domain = normalize_domain(story)
                print(f" {idx:>2}. [{domain}] {synopsis}")
        return

    out_dir = args.output_dir
    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"
    test_path = out_dir / "test.json"
    write_split(splits["train"], train_path)
    write_split(splits["val"], val_path)
    write_split(splits["test"], test_path)

    print(f"Saved splits to {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
