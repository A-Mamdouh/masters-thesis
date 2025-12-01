"""Run heuristic experiments on stories defined in the annotated JSON dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .neural import load_story_neural_policy
from .story import StoryExample
from .story.dataset import load_annotated_stories
from .story_experiments import (
    DEFAULT_BASE_POLICY,
    DEFAULT_POLICIES,
    MAX_EXPLORED_MODELS,
    _configure_logger,
    _policy_from_name,
    _run_policy,
)


def run_annotated_story_experiments(
    stories: Iterable[StoryExample],
    *,
    pool_size: int,
    timeout_ms: int,
    base_policy_name: str,
    comparison_policy_names: Sequence[str],
    output_path: Path | None,
    neural_checkpoint: Path | None = None,
    neural_policy_name: str = "NeuralPolicy",
    neural_drain_size: int = -1,
    neural_device: str | None = None,
) -> None:
    stories = list(stories)
    if not stories:
        raise SystemExit("No stories loaded from the annotated dataset.")
    base_policy = _policy_from_name(base_policy_name)
    comparison_policies = [
        _policy_from_name(name)
        for name in comparison_policy_names
        if name != base_policy.name
    ]
    if neural_checkpoint:
        neural_policy = load_story_neural_policy(
            neural_policy_name,
            neural_checkpoint,
            drain_size=neural_drain_size,
            device=neural_device,
        )
        comparison_policies.append(neural_policy)
    logger = _configure_logger(output_path)
    logger.info(
        "Running annotated experiments for %d stories with base policy '%s' (pool=%d)",
        len(stories),
        base_policy.name,
        pool_size,
    )
    for story in stories:
        logger.info("Results for story: %s", story.synopsis)
        logger.info("  " + "\n  ".join(sentence.text for sentence in story.sentences))
        base_stats = _run_policy(
            story,
            base_policy,
            pool_size,
            timeout_ms,
            max_explored=MAX_EXPLORED_MODELS,
        )
        if not base_stats.model_found or base_stats.model is None:
            logger.error("Base policy '%s' failed to find a model.", base_policy.name)
            continue
        base_model = base_stats.model.clip()
        logger.info(str(base_stats))
        for policy in comparison_policies:
            stats = _run_policy(
                story,
                policy,
                pool_size,
                timeout_ms,
                target_model=base_model,
                target_policy=base_policy,
                max_explored=MAX_EXPLORED_MODELS,
            )
            logger.info(str(stats))
        logger.info("-" * 55)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run heuristic experiments on annotated JSON datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/test.json")],
        help="Annotated story JSON file(s). Default: data/test.json",
    )
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--base-policy", type=str, default=DEFAULT_BASE_POLICY)
    parser.add_argument(
        "--policies",
        type=str,
        nargs="*",
        default=[],
        help="Additional comparison policies (names must be registered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotated_test_results.txt"),
        help="Log output path.",
    )
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        default=None,
        help="Optional neural checkpoint to include as a policy.",
    )
    parser.add_argument(
        "--neural-policy-name",
        type=str,
        default="NeuralPolicy",
    )
    parser.add_argument("--neural-drain-size", type=int, default=-1)
    parser.add_argument("--neural-device", type=str, default=None)
    args = parser.parse_args(argv)
    stories = load_annotated_stories(args.input)
    run_annotated_story_experiments(
        stories,
        pool_size=args.pool_size,
        timeout_ms=args.timeout_ms,
        base_policy_name=args.base_policy,
        comparison_policy_names=args.policies,
        output_path=args.output,
        neural_checkpoint=args.neural_checkpoint,
        neural_policy_name=args.neural_policy_name,
        neural_drain_size=args.neural_drain_size,
        neural_device=args.neural_device,
    )


if __name__ == "__main__":
    main()
