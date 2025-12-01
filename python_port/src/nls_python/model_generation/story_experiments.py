import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

from nls_python.typing_utils import typechecked

from .generators.priority import PriorityModelGenerator
from .policies import (
    BFS_POLICY,
    DFS_POLICY,
    JAVA_CUSTOM_POLICY,
    SearchPolicy,
    get_policy,
    register_policy,
)
from .story import StoryExample, apply_sentence
from .story.dataset import load_annotated_stories
from .tableau_model import TableauModel
from .neural import load_story_neural_policy


DEFAULT_BASE_POLICY = JAVA_CUSTOM_POLICY.name
DEFAULT_POLICIES = (DFS_POLICY.name, BFS_POLICY.name)
DEFAULT_DATASET_PATHS = (Path("data/test.json"),)
MAX_EXPLORED_MODELS = 20_000


@dataclass
class HeuristicStatistics:
    policy: SearchPolicy
    time_ms: float
    explored_models: int
    model_found: bool
    model: TableauModel | None
    candidate_models: List[TableauModel] = field(default_factory=list)
    branch_events: int = 0
    branch_models: int = 0

    def __str__(self) -> str:
        base = "HeuristicStatistics[policy=%s, timeMs=%d, exploredModels=%d, modelFound=%s"
        parts = [
            self.policy.name,
            round(self.time_ms),
            self.explored_models,
            self.model_found,
        ]
        if os.getenv("NLS_DEV_BRANCH_METRICS"):
            base += ", branchEvents=%d, branchModels=%d"
            parts.extend([self.branch_events, self.branch_models])
        base += "]"
        return base % tuple(parts)


def _policy_from_name(name: str) -> SearchPolicy:
    return get_policy(name)


def _load_experiment_stories(
    dataset_paths: Sequence[Path] | None,
) -> List[StoryExample]:
    paths = tuple(dataset_paths or DEFAULT_DATASET_PATHS)
    stories = load_annotated_stories(paths)
    if not stories:
        raise SystemExit(
            f"No stories loaded from dataset paths: {', '.join(str(path) for path in paths)}"
        )
    return stories


def _search_until_target(
    current_model: TableauModel | None,
    generator: PriorityModelGenerator,
    *,
    target_policy: SearchPolicy,
    target_score: float,
    max_additional_explored: int | None = None,
) -> tuple[TableauModel | None, float, bool]:
    """Continue exploring until a model meets the baseline score."""
    tolerance = 1e-9
    extra_time_ms = 0.0
    model = current_model.clip() if current_model else None
    base_explored = generator.get_num_explored_models()
    while model is not None:
        score = target_policy.score(model)
        if score <= target_score + tolerance:
            return model, extra_time_ms, False
        if max_additional_explored is not None:
            current_count = generator.get_num_explored_models()
            if current_count - base_explored >= max_additional_explored:
                return model, extra_time_ms, True
        generator.clear_top_result()
        start = perf_counter()
        next_candidate = generator.generate_model(verbose=False)
        extra_time_ms += (perf_counter() - start) * 1000.0
        if next_candidate is None:
            return None, extra_time_ms, False
        model = next_candidate.clip()
        if max_additional_explored is not None:
            current_count = generator.get_num_explored_models()
            if current_count - base_explored >= max_additional_explored:
                return model, extra_time_ms, True
    return None, extra_time_ms, False


@typechecked
def _run_policy(
    story: StoryExample,
    policy: SearchPolicy,
    pool_size: int,
    timeout_ms: int,
    *,
    target_model: TableauModel | None = None,
    target_policy: SearchPolicy | None = None,
    max_explored: int = MAX_EXPLORED_MODELS,
) -> HeuristicStatistics:
    if (target_model is None) ^ (target_policy is None):
        raise ValueError(
            "target_model and target_policy must both be provided or both be None."
        )
    models: List[TableauModel] = [TableauModel()]
    models[0].set_story_key(story.key)
    total_time = 0.0
    total_explored = 0
    final_model: TableauModel | None = None
    total_branch_events = 0
    total_branch_models = 0
    final_candidates: List[TableauModel] = []
    final_generator: PriorityModelGenerator | None = None
    def exceeded_cap() -> bool:
        return max_explored > 0 and total_explored >= max_explored

    def make_stats(
        *,
        model: TableauModel | None,
        candidates: Sequence[TableauModel] | None = None,
        aborted: bool = False,
    ) -> HeuristicStatistics:
        candidate_models: List[TableauModel] = (
            [candidate.clip() for candidate in candidates] if candidates else []
        )
        if aborted:
            model = None
            candidate_models = []
        return HeuristicStatistics(
            policy=policy,
            time_ms=total_time,
            explored_models=total_explored,
            model_found=bool(model) and not aborted,
            model=model,
            branch_events=total_branch_events,
            branch_models=total_branch_models,
            candidate_models=candidate_models,
        )

    for index, sentence in enumerate(story.sentences):
        seeded_models: List[TableauModel] = []
        for model in models:
            clone = model.clip()
            apply_sentence(clone, sentence, index)
            seeded_models.append(clone)
        if not seeded_models:
            return make_stats(model=None)
        generator = PriorityModelGenerator(
            initial_model=seeded_models[0],
            initial_models=seeded_models[1:],
            cost_function=policy.cost_function,
        )
        generator.set_timeout(timeout_ms)
        start = perf_counter()
        candidate = generator.generate_model(verbose=False)
        total_time += (perf_counter() - start) * 1000.0
        if candidate is None:
            generator.close()
            return make_stats(model=None)
        final_model = candidate.clip()
        models = generator.collect_leaves()
        is_last_sentence = index == len(story.sentences) - 1
        if is_last_sentence:
            final_generator = generator
        else:
            total_explored += generator.get_num_explored_models()
            branch_events, branch_models = generator.get_branch_statistics()
            total_branch_events += branch_events
            total_branch_models += branch_models
            generator.close()
            if exceeded_cap():
                return make_stats(model=final_model, candidates=models)
            if not models:
                models = [final_model.clip()]
        if exceeded_cap():
            return make_stats(model=None, aborted=True)
    if final_generator is None:
        return make_stats(model=None)
    if models:
        final_candidates = [model.clip() for model in models]
    capped_during_search = False
    if target_model and target_policy and (max_explored <= 0 or not exceeded_cap()):
        remaining_budget = (
            None if max_explored <= 0 else max(0, max_explored - total_explored)
        )
        if remaining_budget == 0:
            capped_during_search = True
        else:
            target_score = target_policy.score(target_model)
            final_model, extra_time, capped_during_search = _search_until_target(
                final_model,
                final_generator,
                target_policy=target_policy,
                target_score=target_score,
                max_additional_explored=remaining_budget,
            )
            total_time += extra_time
    elif target_model and target_policy:
        target_score = target_policy.score(target_model)
        final_model, extra_time, capped_during_search = _search_until_target(
            final_model,
            final_generator,
            target_policy=target_policy,
            target_score=target_score,
        )
        total_time += extra_time
    total_explored += final_generator.get_num_explored_models()
    branch_events, branch_models = final_generator.get_branch_statistics()
    total_branch_events += branch_events
    total_branch_models += branch_models
    final_generator.close()
    if exceeded_cap() or capped_during_search:
        return make_stats(model=None, aborted=True)
    return make_stats(model=final_model, candidates=final_candidates)


def _configure_logger(output_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("story_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if output_path:
        file_handler = logging.FileHandler(output_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def run_story_experiments(
    *,
    pool_size: int,
    timeout_ms: int,
    base_policy_name: str = DEFAULT_BASE_POLICY,
    comparison_policy_names: Sequence[str] = DEFAULT_POLICIES,
    dataset_paths: Sequence[Path] | None = None,
    output_path: Path | None = None,
    neural_policy: SearchPolicy | None = None,
) -> None:
    base_policy = _policy_from_name(base_policy_name)
    if neural_policy:
        register_policy(neural_policy, override=True)
    comparison_policies = [
        _policy_from_name(name)
        for name in comparison_policy_names
        if name != base_policy.name
    ]
    logger = _configure_logger(output_path)
    stories = _load_experiment_stories(dataset_paths)
    dataset_summary = ", ".join(
        str(path) for path in (dataset_paths or DEFAULT_DATASET_PATHS)
    )
    logger.info(
        "Running experiments for %d stories (dataset: %s) with base policy '%s' and pool size %d",
        len(stories),
        dataset_summary,
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
        )
        if not base_stats.model_found or base_stats.model is None:
            logger.error("Base policy '%s' failed to find a model.", base_policy.name)
            continue
        base_model = base_stats.model.clip()
        logger.info(str(base_stats))
        # if base_stats.model:
        #     logger.info("Base model snapshot:\n%s", base_stats.model.create_string())
        for policy in comparison_policies:
            stats = _run_policy(
                story,
                policy,
                pool_size,
                timeout_ms,
                target_model=base_model,
                target_policy=base_policy,
            )
            logger.info(str(stats))
            # if stats.model:
            #     logger.info(
            #         "%s snapshot:\n%s", policy.name, stats.model.create_string()
            #     )
        logger.info("-" * 55)


def _default_output_path(pool_size: int, timeout_ms: int) -> Path:
    return Path(f"test_results_{pool_size}_workers_{timeout_ms}ms.txt")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run heuristic experiments on story bank."
    )
    parser.add_argument(
        "--pool-size", type=int, default=12, help="Worker pool size (default: 12)."
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=200,
        help="Timeout per generator invocation (ms).",
    )
    parser.add_argument(
        "--base-policy",
        type=str,
        default=DEFAULT_BASE_POLICY,
        help=f"Base policy name (default: {DEFAULT_BASE_POLICY}).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="*",
        default=list(DEFAULT_POLICIES),
        help="Policies to compare (default: DFS BFS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional log file path. Defaults to test_results_{pool}_workers_{timeout}ms.txt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[Path("data/test.json")],
        help="Annotated dataset JSON file(s) to load (default: data/test.json).",
    )
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        default=None,
        help="Optional path to a neural policy checkpoint.",
    )
    parser.add_argument(
        "--neural-policy-name",
        type=str,
        default="NeuralPolicy",
        help="Name to register the neural policy under.",
    )
    parser.add_argument(
        "--neural-drain-size",
        type=int,
        default=-1,
        help="Drain size to use for the neural policy.",
    )
    parser.add_argument(
        "--neural-device",
        type=str,
        default=None,
        help="Device override when loading the neural checkpoint (e.g., cpu).",
    )
    args = parser.parse_args(argv)
    output_path = args.output or _default_output_path(args.pool_size, args.timeout_ms)
    neural_policy = None
    if args.neural_checkpoint:
        neural_policy = load_story_neural_policy(
            args.neural_policy_name,
            args.neural_checkpoint,
            drain_size=args.neural_drain_size,
            device=args.neural_device,
        )
    run_story_experiments(
        pool_size=args.pool_size,
        timeout_ms=args.timeout_ms,
        base_policy_name=args.base_policy,
        comparison_policy_names=args.policies,
        dataset_paths=args.dataset,
        output_path=output_path,
        neural_policy=neural_policy,
    )


if __name__ == "__main__":
    main()
