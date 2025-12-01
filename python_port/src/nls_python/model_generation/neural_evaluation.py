import logging
import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")  # Avoid GUI requirements during CLI/tests.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

from nls_python.typing_utils import typechecked

from .generators.priority import PriorityModelGenerator
from .policies import JAVA_CUSTOM_POLICY, SearchPolicy, get_policy
from .story import (
    LexicographicStoryOrderer,
    StoryOrderer,
    StoryExample,
    apply_sentence,
    order_story_models,
)
from .story.dataset import DEFAULT_DATASET_PATHS, load_annotated_stories
from .tableau_model import TableauModel
from .neural import load_story_neural_policy

logger = logging.getLogger(__name__)

def _drain_size(policy: SearchPolicy) -> int | None:
    if policy.drain_size <= 0:
        return None
    return policy.drain_size


@typechecked
def _replay_story(
    story: StoryExample, policy: SearchPolicy, pool_size: int, timeout_ms: int
) -> Tuple[Optional[TableauModel], List[TableauModel]]:
    models: List[TableauModel] = [TableauModel()]
    models[0].set_story_key(story.key)
    final_model: TableauModel | None = None
    for index, sentence in enumerate(story.sentences):
        seeded_models: List[TableauModel] = []
        for model in models:
            clone = model.clip()
            apply_sentence(clone, sentence, index)
            seeded_models.append(clone)
        if not seeded_models:
            return None, []
        generator = PriorityModelGenerator(
            initial_model=seeded_models[0],
            initial_models=seeded_models[1:],
            cost_function=policy.cost_function,
        )
        generator.set_timeout(timeout_ms)
        candidate = generator.generate_model(verbose=False)
        leaves = generator.collect_leaves()
        generator.close()
        if candidate:
            final_model = candidate.clip()
        models = leaves or ([candidate.clip()] if candidate else [])
        if not models:
            break
    return final_model, models


def _spearman(
    canonical: Sequence[TableauModel], evaluated: Sequence[TableauModel]
) -> float:
    n = len(canonical)
    if n <= 1:
        return 1.0
    ranks = {model: idx for idx, model in enumerate(canonical)}
    diff_sq = 0
    for idx, model in enumerate(evaluated):
        diff = ranks.get(model, idx) - idx
        diff_sq += diff * diff
    return 1.0 - (6.0 * diff_sq) / (n * (n * n - 1))


def evaluate_neural_policy(
    stories: Iterable[StoryExample],
    *,
    policy: SearchPolicy,
    orderer: StoryOrderer,
    pool_size: int,
    timeout_ms: int,
    top_k: int = 1,
) -> Tuple[Dict[str, float], float, float]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    correlations: Dict[str, float] = {}
    found: List[float] = []
    top_hits: List[float] = []
    for story in stories:
        final_model, leaves = _replay_story(story, policy, pool_size, timeout_ms)
        found.append(1.0 if final_model is not None else 0.0)
        if leaves:
            canonical = order_story_models(orderer, leaves)
            scored = sorted(leaves, key=lambda model: policy.score(model))
            correlations[story.key] = _spearman(canonical, scored)
            target_model = canonical[0]
            limit = min(top_k, len(scored))
            top_slice = scored[:limit]
            hit = 1.0 if target_model in top_slice else 0.0
            top_hits.append(hit)
        else:
            correlations[story.key] = 0.0
            top_hits.append(0.0)
    return (
        correlations,
        mean(found) if found else 0.0,
        mean(top_hits) if top_hits else 0.0,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate neural policy across the story bank."
    )
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        required=True,
        help="Path to neural checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=[Path("data/test.json")],
        help="Annotated story JSON files (default: data/test.json).",
    )
    parser.add_argument(
        "--story-keys",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of story keys to evaluate.",
    )
    parser.add_argument(
        "--orderer", type=str, default="lexicographic", choices=("lexicographic",)
    )
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--policy-name", type=str, default="NeuralEval")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Compute top-k accuracy for this cutoff (default: 5).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory to store evaluation plots.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    orderer = LexicographicStoryOrderer()
    policy = load_story_neural_policy(
        args.policy_name,
        args.neural_checkpoint,
        drain_size=-1,
        device=args.device,
    )
    stories = load_annotated_stories(tuple(args.datasets))
    if args.story_keys:
        key_filter = set(args.story_keys)
        stories = [story for story in stories if story.key in key_filter]
    if not stories:
        raise SystemExit("No stories available for evaluation.")
    correlations, found_rate, topk_accuracy = evaluate_neural_policy(
        stories,
        policy=policy,
        orderer=orderer,
        pool_size=args.pool_size,
        timeout_ms=args.timeout_ms,
        top_k=args.top_k,
    )
    base_policy = get_policy(JAVA_CUSTOM_POLICY.name)
    _, base_found, base_topk = evaluate_neural_policy(
        stories,
        policy=base_policy,
        orderer=orderer,
        pool_size=args.pool_size,
        timeout_ms=args.timeout_ms,
        top_k=args.top_k,
    )
    print("Story-wise Spearman correlations:")
    for story_key, corr in correlations.items():
        print(f"  {story_key}: {corr:.3f}")
    avg_corr = mean(correlations.values()) if correlations else 0.0
    logger.info("Average Spearman: %.3f", avg_corr)
    logger.info("Neural model-found rate: %.2f%%", found_rate * 100)
    logger.info("CustomJava model-found rate: %.2f%%", base_found * 100)
    logger.info(
        "Neural top-%d accuracy: %.2f%%", args.top_k, topk_accuracy * 100
    )
    logger.info(
        "CustomJava top-%d accuracy: %.2f%%", args.top_k, base_topk * 100
    )
    if args.plot_dir:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for plotting. Install it or omit --plot-dir."
            )
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(correlations)), list(correlations.values()))
        plt.xticks(range(len(correlations)), list(correlations.keys()), rotation=45)
        plt.ylabel("Spearman")
        plt.title("Per-story Spearman correlations")
        plt.tight_layout()
        plt.savefig(args.plot_dir / "spearman.png")
        plt.close()

        plt.figure()
        plt.bar(
            ["Neural", "CustomJava"],
            [found_rate, base_found],
            color=["C0", "C1"],
        )
        plt.ylabel("Model-found rate")
        plt.ylim(0, 1)
        plt.title("Model-found comparison")
        plt.tight_layout()
        plt.savefig(args.plot_dir / "model_found.png")
        plt.close()

        plt.figure()
        plt.bar(
            ["Neural", "CustomJava"],
            [topk_accuracy, base_topk],
            color=["C2", "C3"],
        )
        plt.ylabel(f"Top-{args.top_k} accuracy")
        plt.ylim(0, 1)
        plt.title("Top-k accuracy comparison")
        plt.tight_layout()
        plt.savefig(args.plot_dir / "topk_accuracy.png")
        plt.close()


if __name__ == "__main__":
    main()
