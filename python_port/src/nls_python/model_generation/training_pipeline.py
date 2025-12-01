from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from nls_python.typing_utils import typechecked

from .neural import load_story_neural_policy
from .neural_evaluation import evaluate_neural_policy
from .neural_training import train_from_dataset
from .story import LexicographicStoryOrderer, StoryOrderer
from .story.dataset import DEFAULT_DATASET_PATHS, load_annotated_stories

logger = logging.getLogger(__name__)


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"Dataset {path} is empty")
    return examples


def _write_examples(path: Path, examples: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")


@typechecked
def split_dataset(
    dataset_path: Path,
    train_output: Path,
    val_output: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[int, int, int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    examples = _load_examples(dataset_path)
    rng = random.Random(seed)  # type: ignore[name-defined]
    rng.shuffle(examples)
    split_index = max(1, int(len(examples) * (1 - val_ratio)))
    train_examples = examples[:split_index]
    val_examples = examples[split_index:]
    if not train_examples or not val_examples:
        raise ValueError("Split produced an empty train or validation set")
    _write_examples(train_output, train_examples)
    _write_examples(val_output, val_examples)
    return len(examples), len(train_examples), len(val_examples)


@typechecked
def run_training_pipeline(
    *,
    dataset_path: Path,
    train_output: Path,
    val_output: Path,
    checkpoint_path: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    epochs: int = 3,
    lr: float = 1e-4,
    device: str = "cpu",
    log_dir: Path | None = None,
    story_datasets: Sequence[Path] | None = None,
    story_keys: Sequence[str] | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    total, train_count, val_count = split_dataset(
        dataset_path,
        train_output,
        val_output,
        val_ratio=val_ratio,
        seed=seed,
    )
    logger.info(
        "Split dataset '%s' into %d train / %d val examples (total=%d)",
        dataset_path,
        train_count,
        val_count,
        total,
    )
    train_from_dataset(
        train_output,
        output_path=checkpoint_path,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=device,
        log_dir=log_dir,
    )
    policy = load_story_neural_policy(
        "TrainingPipelinePolicy",
        checkpoint_path,
        device=device,
    )
    orderer: StoryOrderer = LexicographicStoryOrderer()
    eval_paths = tuple(story_datasets or (Path("data/test.json"),))
    eval_stories = load_annotated_stories(eval_paths)
    if story_keys:
        key_filter = set(story_keys)
        eval_stories = [story for story in eval_stories if story.key in key_filter]
    if not eval_stories:
        joined = ", ".join(str(path) for path in eval_paths)
        raise ValueError(f"No annotated stories available for evaluation from: {joined}")
    correlations, found, topk = evaluate_neural_policy(
        eval_stories,
        policy=policy,
        orderer=orderer,
        pool_size=4,
        timeout_ms=200,
        top_k=top_k,
    )
    summary = {
        "train_examples": train_count,
        "val_examples": val_count,
        "stories_evaluated": tuple(story.key for story in eval_stories),
        "spearman_by_story": correlations,
        "model_found_rate": found,
        "topk_accuracy": topk,
        "topk_cutoff": top_k,
    }
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = log_dir / "evaluation.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    logger.info("Evaluation summary: %s", summary)
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end training pipeline helper.")
    parser.add_argument("--dataset", type=Path, required=True, help="Input JSONL dataset.")
    parser.add_argument("--train-output", type=Path, required=True, help="Path to write the train split.")
    parser.add_argument("--val-output", type=Path, required=True, help="Path to write the validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Destination checkpoint path.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu/cuda).")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional directory for training/eval logs.")
    parser.add_argument(
        "--story-datasets",
        type=Path,
        nargs="+",
        default=[Path("data/test.json")],
        help="Annotated story JSON files for evaluation (default: data/test.json).",
    )
    parser.add_argument(
        "--story-keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of story keys (filtered from the provided datasets).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff for evaluation accuracy (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run_training_pipeline(
        dataset_path=args.dataset,
        train_output=args.train_output,
        val_output=args.val_output,
        checkpoint_path=args.checkpoint,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        log_dir=args.log_dir,
        story_datasets=args.story_datasets,
        story_keys=args.story_keys,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
