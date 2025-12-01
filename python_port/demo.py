"""Minimal demo script to replay an annotated story with a chosen policy (symbolic or neural).

Usage examples (from repository root):
  uv run python demo.py --dataset data/extra_test.json --base-policy CustomJava
  uv run python demo.py --dataset data/extra_test.json --neural-checkpoint experiments/policies/CustomJava.pt --neural-device cpu
"""



import argparse
import json
import sys
from pathlib import Path
from typing import List


# Allow running from the repo without installing the package.
def _ensure_path() -> None:
    try:  # pragma: no cover
        import nls_python  # noqa: F401
        return
    except ImportError:
        repo_root = Path(__file__).resolve().parent
        sys.path.append(str(repo_root / "src"))


_ensure_path()

from nls_python.model_generation.generators.priority import PriorityModelGenerator
from nls_python.model_generation.neural import load_story_neural_policy
from nls_python.model_generation.policies import JAVA_CUSTOM_POLICY, get_policy
from nls_python.model_generation.story import StoryExample, StorySentence, apply_sentence
from nls_python.model_generation.tableau_model import TableauModel


def _load_stories(path: Path) -> List[StoryExample]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a list of stories: {path}")
    stories: List[StoryExample] = []
    for idx, raw in enumerate(payload):
        sentences = raw.get("sentences") or []
        story_key = raw.get("key") or f"{path.stem}_{idx}"
        synopsis = raw.get("synopsis") or raw.get("domain") or story_key
        parsed_sentences: List[StorySentence] = []
        for s_idx, sentence in enumerate(sentences):
            annotation = sentence.get("annotation") or {}
            roles = annotation.get("roles") or {}
            anaphora = annotation.get("anaphora") or {}
            semantic_types = annotation.get("semantic_types") or {}
            # Ensure every role value is either resolved via anaphora or has a semantic type
            for value in roles.values():
                if value not in anaphora and value not in semantic_types:
                    semantic_types[value] = "unknown"
            parsed_sentences.append(
                StorySentence(
                    verb=str(annotation.get("verb", "")),
                    roles=roles,
                    negated=bool(annotation.get("negated", False)),
                    text=str(sentence.get("text", "")),
                    anaphora=anaphora,
                    semantic_types=semantic_types,
                )
            )
        stories.append(
            StoryExample(
                key=story_key,
                synopsis=synopsis,
                sentences=parsed_sentences,
            )
        )
    return stories


def _run_story(story: StoryExample, policy, timeout_ms: int) -> TableauModel | None:
    models: List[TableauModel] = [TableauModel()]
    models[0].set_story_key(story.key)
    for index, sentence in enumerate(story.sentences):
        seeded_models: List[TableauModel] = []
        for model in models:
            clone = model.clip()
            apply_sentence(clone, sentence, index)
            seeded_models.append(clone)
        if not seeded_models:
            return None
        generator = PriorityModelGenerator(
            initial_model=seeded_models[0],
            initial_models=seeded_models[1:],
            cost_function=policy.cost_function,
        )
        generator.set_timeout(timeout_ms)
        candidate = generator.generate_model(verbose=False)
        models = generator.collect_leaves() or ([candidate.clip()] if candidate else [])
        generator.close()
        if not models:
            return None
    return models[0] if models else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo: run a story through a policy.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/extra_test.json"),
        help="Path to annotated story JSON (list of stories).",
    )
    parser.add_argument(
        "--base-policy",
        type=str,
        default=JAVA_CUSTOM_POLICY.name,
        help="Symbolic policy name to use when no neural checkpoint is provided.",
    )
    parser.add_argument(
        "--neural-checkpoint",
        type=Path,
        default=None,
        help="Optional neural checkpoint to load as the policy.",
    )
    parser.add_argument(
        "--neural-name",
        type=str,
        default="NeuralPolicy",
        help="Name for the neural policy when loaded.",
    )
    parser.add_argument(
        "--neural-drain-size",
        type=int,
        default=1,
        help="Drain size for the neural policy (<=0 means unlimited).",
    )
    parser.add_argument(
        "--neural-device",
        type=str,
        default="cpu",
        help="Device override for loading the neural checkpoint (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=600,
        help="Timeout per generator invocation (milliseconds).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stories = _load_stories(args.dataset)
    if not stories:
        raise SystemExit(f"No stories found in {args.dataset}")
    if args.neural_checkpoint:
        policy = load_story_neural_policy(
            args.neural_name,
            args.neural_checkpoint,
            drain_size=args.neural_drain_size,
            device=args.neural_device,
        )
        print(f"Using neural policy from {args.neural_checkpoint}")
    else:
        policy = get_policy(args.base_policy)
        print(f"Using symbolic policy '{policy.name}'")
    for story in stories:
        print(f"\nStory: {story.synopsis} (key={story.key})")
        for sentence in story.sentences:
            print(f"  - {sentence.text}")
        model = _run_story(story, policy, timeout_ms=args.timeout_ms)
        if model:
            print("\nFinal tableau model:\n")
            print(model.create_string())
        else:
            print("\nNo model found for this story.")


if __name__ == "__main__":
    main()
