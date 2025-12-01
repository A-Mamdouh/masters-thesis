from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from nls_python.typing_utils import typechecked

from .neural import load_story_neural_policy
from .neural_evaluation import evaluate_neural_policy
from .neural_training import train_from_dataset
from .story import (
    LexicographicStoryOrderer,
    OllamaStoryOrderer,
    StoryOrderer,
)
from .story.dataset import DEFAULT_DATASET_PATHS, load_annotated_stories
from .training_data import write_training_dataset

logger = logging.getLogger(__name__)

DEFAULT_POLICIES: Sequence[str] = ("CustomJava", "DepthBiased", "BFS", "DFS")


def _orderer_from_name(name: str) -> StoryOrderer:
    normalized = name.lower()
    if normalized == "lexicographic":
        return LexicographicStoryOrderer()
    if normalized == "ollama":
        return OllamaStoryOrderer()
    raise ValueError(f"Unknown orderer '{name}'. Available: lexicographic, ollama")


@dataclass(frozen=True)
class ExperimentResult:
    policy: str
    dataset_examples: int
    checkpoint_path: Path
    metrics: Dict[str, float]
    correlations: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@typechecked
def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle if _.strip())


@typechecked
def run_policy_experiment(
    policy: str,
    *,
    dataset_paths: Sequence[Path],
    orderer: StoryOrderer,
    output_dir: Path,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pool_size: int,
    timeout_ms: int,
    top_k: int,
    neural_drain_size: int,
    overwrite: bool,
) -> ExperimentResult:
    logger.info(f"{dataset_paths}")
    stories = load_annotated_stories(dataset_paths)
    if not stories:
        joined = ", ".join(str(path) for path in dataset_paths)
        raise ValueError(f"No annotated stories found in: {joined}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{policy}.jsonl"
    checkpoint_path = output_dir / f"{policy}.pt"
    log_dir = output_dir / f"{policy}_logs"

    if overwrite or not dataset_path.exists():
        logger.info("Generating dataset for %s -> %s", policy, dataset_path)
        count, _, _ = write_training_dataset(
            dataset_path,
            orderer=orderer,
            stories=stories,
            policies=(policy,),
            pool_size=pool_size,
            timeout_ms=timeout_ms,
        )
    else:
        logger.info("Reusing existing dataset for %s at %s", policy, dataset_path)
        count = _line_count(dataset_path)

    if overwrite or not checkpoint_path.exists():
        logger.info("Training neural mimic for %s", policy)
        train_from_dataset(
            dataset_path,
            output_path=checkpoint_path,
            epochs=epochs,
            lr=lr,
            seed=seed,
            device=device,
            log_dir=log_dir,
        )
    else:
        logger.info("Reusing existing checkpoint for %s at %s", policy, checkpoint_path)

    neural_policy = load_story_neural_policy(
        f"{policy}_Neural",
        checkpoint_path,
        drain_size=neural_drain_size,
        device=device,
    )
    correlations, found, topk = evaluate_neural_policy(
        stories,
        policy=neural_policy,
        orderer=orderer,
        pool_size=pool_size,
        timeout_ms=timeout_ms,
        top_k=top_k,
    )
    metrics = {
        "model_found_rate": found,
        "topk_accuracy": topk,
    }
    avg_corr = (
        sum(correlations.values()) / len(correlations) if correlations else 0.0
    )
    metrics["avg_spearman"] = avg_corr
    return ExperimentResult(
        policy=policy,
        dataset_examples=count,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        correlations=correlations,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch training/evaluation for symbolic policy mimics."
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="*",
        default=list(DEFAULT_POLICIES),
        help=f"Policies to distill (default: {', '.join(DEFAULT_POLICIES)}).",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=list(DEFAULT_DATASET_PATHS),
        help="Annotated story JSON files to replay (default: data/train.json data/val.json data/test.json).",
    )
    parser.add_argument(
        "--orderer",
        type=str,
        default="lexicographic",
        choices=("lexicographic", "ollama"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/policies"),
        help="Directory to store datasets, checkpoints, and logs.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neural-drain-size", type=int, default=-1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate datasets/checkpoints even if they already exist.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write aggregated JSON summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    orderer = _orderer_from_name(args.orderer)
    results: List[ExperimentResult] = []
    for policy in args.policies:
        logger.info("=== Running experiment for %s ===", policy)
        result = run_policy_experiment(
            policy,
            dataset_paths=tuple(args.datasets),
            orderer=orderer,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            pool_size=args.pool_size,
            timeout_ms=args.timeout_ms,
            top_k=args.top_k,
            neural_drain_size=args.neural_drain_size,
            overwrite=args.overwrite,
        )
        logger.info(
            "%s: dataset=%d examples, found=%.2f%%, top-%d=%.2f%%, avg Spearman=%.3f",
            policy,
            result.dataset_examples,
            result.metrics["model_found_rate"] * 100,
            args.top_k,
            result.metrics["topk_accuracy"] * 100,
            result.metrics["avg_spearman"],
        )
        results.append(result)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump([result.to_dict() for result in results], handle, indent=2)
        logger.info("Wrote summary to %s", args.summary)


if __name__ == "__main__":
    main()
