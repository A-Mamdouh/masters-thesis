import argparse
import json
import random
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TextIO

from nls_python.typing_utils import typechecked

from .generators.priority import PriorityModelGenerator
from .heuristics import ModelFeatures, basic_model_features
from .policies import JAVA_CUSTOM_POLICY, get_policy
from .story import (
    LexicographicStoryOrderer,
    OllamaStoryOrderer,
    StoryOrderer,
    StoryExample,
    apply_sentence,
    order_story_models,
)
from .story.dataset import DEFAULT_DATASET_PATHS, load_annotated_stories
from .tableau_model import TableauModel


logger = logging.getLogger(__name__)

@typechecked
@dataclass(frozen=True)
class TrainingExample:
    story_key: str
    sentence_index: int
    policy: str
    model_id: int
    rank: int
    total_models: int
    rank_fraction: float
    rank_from_top: int
    best_rank: int
    narrative: str
    features: ModelFeatures
    narrative_steps: List[Dict[str, Any]]
    num_rule_applications: int
    num_extensions: int
    num_pending_branches: int

    def to_json(self) -> str:
        payload = {
            "story_key": self.story_key,
            "sentence_index": self.sentence_index,
            "policy": self.policy,
            "model_id": self.model_id,
            "rank": self.rank,
            "total_models": self.total_models,
            "rank_fraction": self.rank_fraction,
            "rank_from_top": self.rank_from_top,
            "best_rank": self.best_rank,
            "narrative": self.narrative,
            "features": asdict(self.features),
            "narrative_steps": self.narrative_steps,
            "num_rule_applications": self.num_rule_applications,
            "num_extensions": self.num_extensions,
            "num_pending_branches": self.num_pending_branches,
        }
        return json.dumps(payload, ensure_ascii=False)


def _rank_models(
    models: Sequence[TableauModel],
    orderer: StoryOrderer,
) -> List[TableauModel]:
    if not models:
        return []
    return order_story_models(orderer, models)


def _sentence_examples(
    story: StoryExample,
    policy_name: str,
    orderer: StoryOrderer,
    *,
    pool_size: int,
    timeout_ms: int,
) -> Iterator[TrainingExample]:
    policy = get_policy(policy_name)
    models: List[TableauModel] = [TableauModel()]
    models[0].set_story_key(story.key)
    for index, sentence in enumerate(story.sentences):
        logger.info(f"Sentence: {index + 1} / {len(story.sentences)} - {sentence.text}")
        start = time.time()
        seeded_models: List[TableauModel] = []
        for model in models:
            clone = model.clip()
            apply_sentence(clone, sentence, index)
            seeded_models.append(clone)
        if not seeded_models:
            break
        generator = PriorityModelGenerator(
            initial_model=seeded_models[0],
            initial_models=seeded_models[1:],
            cost_function=policy.cost_function,
        )
        generator.set_timeout(timeout_ms)
        candidate = generator.generate_model(verbose=False)
        leaves = generator.collect_leaves()
        generator.close()
        logger.info(f"--Time taken: {time.time() - start:.2f}s--")
        if not leaves and candidate:
            leaves = [candidate.clip()]
        ranked_models = _rank_models(leaves, orderer)
        total = len(ranked_models)
        for rank, model in enumerate(ranked_models):
            features = basic_model_features(model)
            narrative = model.get_narrative().story_text()
            extensions = model.get_extensions()
            pending_branches = sum(len(branches) for branches in extensions.values())
            steps = [
                {
                    "index": step.index,
                    "verb": step.verb,
                    "roles": dict(step.roles),
                    "negated": step.negated,
                    "text": step.text,
                    "anaphora": dict(step.anaphora),
                    "semantic_types": dict(step.semantic_types),
                }
                for step in model.get_narrative().steps
            ]
            rank_fraction = (
                0.0 if total <= 1 else rank / (total - 1)
            )
            yield TrainingExample(
                story_key=story.key,
                sentence_index=index,
                policy=policy_name,
                model_id=model.id,
                rank=rank,
                total_models=total,
                rank_fraction=float(rank_fraction),
                rank_from_top=rank + 1,
                best_rank=1,
                narrative=narrative,
                features=features,
                narrative_steps=steps,
                num_rule_applications=len(model.get_rule_applications()),
                num_extensions=len(extensions),
                num_pending_branches=pending_branches,
            )
        models = leaves or ([candidate.clip()] if candidate else [])
        if not models:
            break


def iter_training_examples(
    *,
    orderer: StoryOrderer,
    stories: Sequence[StoryExample],
    policies: Sequence[str] | None = None,
    pool_size: int = 4,
    timeout_ms: int = 200,
) -> Iterator[TrainingExample]:
    default_policy = JAVA_CUSTOM_POLICY.name
    policy_names = tuple(policies or (default_policy,))
    for story_id, story in enumerate(stories, 1):
        for policy_id, policy_name in enumerate(policy_names):
            logger.info(f"Yielding Story {story_id} / {len(stories)} - Policy: {policy_name} {policy_id+1} / {len(policy_names)}")
            yield from _sentence_examples(
                story,
                policy_name,
                orderer,
                pool_size=pool_size,
                timeout_ms=timeout_ms,
            )


def _resolve_stories(
    stories: Sequence[StoryExample] | None,
    dataset_paths: Sequence[Path] | None,
) -> Sequence[StoryExample]:
    if stories:
        return tuple(stories)
    paths = tuple(dataset_paths or DEFAULT_DATASET_PATHS)
    loaded = load_annotated_stories(paths)
    if not loaded:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"No annotated stories found in: {joined}")
    return tuple(loaded)


def _orderer_from_name(name: str) -> StoryOrderer:
    normalized = name.lower()
    if normalized == "lexicographic":
        return LexicographicStoryOrderer()
    if normalized == "ollama":
        return OllamaStoryOrderer()
    raise ValueError(f"Unknown orderer '{name}'. Available: lexicographic, ollama")


def _open_optional(path: Optional[Path]) -> Optional[TextIO]:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_training_dataset(
    output_path: Path | None,
    *,
    orderer: StoryOrderer,
    stories: Sequence[StoryExample] | None = None,
    dataset_paths: Sequence[Path] | None = None,
    policies: Sequence[str] | None = None,
    pool_size: int = 4,
    timeout_ms: int = 200,
    train_output: Path | None = None,
    val_output: Path | None = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> tuple[int, int, int]:
    rng = random.Random(split_seed)
    story_collection = _resolve_stories(stories, dataset_paths)
    if not story_collection:
        raise ValueError("No stories available for dataset generation.")
    count = train_count = val_count = 0
    base_handle = _open_optional(output_path)
    train_handle = _open_optional(train_output)
    val_handle = _open_optional(val_output)
    try:
        for example in iter_training_examples(
            orderer=orderer,
            stories=story_collection,
            policies=policies,
            pool_size=pool_size,
            timeout_ms=timeout_ms,
        ):  
            line = example.to_json() + "\n"
            if base_handle:
                base_handle.write(line)
            if train_handle or val_handle:
                target_val = rng.random() < val_ratio
                if target_val and val_handle:
                    val_handle.write(line)
                    val_count += 1
                elif train_handle:
                    train_handle.write(line)
                    train_count += 1
                elif val_handle:
                    val_handle.write(line)
                    val_count += 1
            count += 1
    finally:
        if base_handle:
            base_handle.close()
        if train_handle:
            train_handle.close()
        if val_handle:
            val_handle.close()
    return count, train_count, val_count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data for neural heuristics."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination JSONL file."
    )
    parser.add_argument(
        "--orderer",
        type=str,
        default="lexicographic",
        choices=("lexicographic", "ollama"),
        help="Story ordering backend.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="+",
        default=list(DEFAULT_DATASET_PATHS),
        help="Annotated story JSON files to replay (default: data/train.json data/val.json data/test.json).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="*",
        default=[JAVA_CUSTOM_POLICY.name],
        help=f"Policies to replay for data generation (default: {JAVA_CUSTOM_POLICY.name}).",
    )
    parser.add_argument("--pool-size", type=int, default=4, help="Worker pool size.")
    parser.add_argument(
        "--timeout-ms", type=int, default=200, help="Generator timeout per step."
    )
    args = parser.parse_args(argv)
    orderer = _orderer_from_name(args.orderer)
    written = write_training_dataset(
        args.output,
        orderer=orderer,
        dataset_paths=args.datasets,
        policies=args.policies,
        pool_size=args.pool_size,
        timeout_ms=args.timeout_ms,
    )
    print(f"Wrote {written} training examples to {args.output}")


if __name__ == "__main__":
    main()
