#!/usr/bin/env python3
"""Quick end-to-end sanity check for a single policy on one annotated story."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from nls_python.model_generation.generators.priority import PriorityModelGenerator
from nls_python.model_generation import apply_sentence
from nls_python.model_generation.policies import SearchPolicy, get_policy
from nls_python.model_generation.story import StoryExample
from nls_python.model_generation.story.dataset import load_annotated_stories
from nls_python.model_generation.tableau_model import TableauModel


def _run_full_story(
    story: StoryExample,
    policy: SearchPolicy,
    *,
    timeout_ms: int,
) -> tuple[TableauModel | None, int, float]:
    models = [TableauModel()]
    models[0].set_story_key(story.key)
    total_explored = 0
    total_time_ms = 0.0
    final_model: TableauModel | None = None
    start = perf_counter()
    print("Num Sentences:", len(story.sentences))
    for index, sentence in enumerate(story.sentences):
        print(f"[{(perf_counter() - start):.2f}s] Sentence {index}: {sentence.text} [{total_explored} models]")
        seeded: list[TableauModel] = []
        for model in models:
            clone = model.clip()
            apply_sentence(clone, sentence, index)
            seeded.append(clone)
        if not seeded:
            return None, total_explored, total_time_ms
        generator = PriorityModelGenerator(
            initial_model=seeded[0],
            initial_models=seeded[1:],
            cost_function=policy.cost_function,
        )
        generator.set_timeout(100000)
        start = perf_counter()
        candidate = generator.generate_model(verbose=False)
        total_explored += generator.get_num_explored_models()
        total_time_ms += (perf_counter() - start) * 1000.0
        if candidate:
            final_model = candidate.clip()
        models = generator.collect_leaves()
        generator.close()
        if not models and final_model is not None:
            models = [final_model.clip()]
        if not models:
            return None, total_explored, total_time_ms
    # print(final_model)
    # models = [model for model in models if not model.is_complete()]
    # model = None
    # while models:
    #     generator = PriorityModelGenerator(
    #         initial_model=models[0],
    #         initial_models=models[1:],
    #         cost_function=policy.cost_function,
    #     )
    #     generator.set_timeout(100000)
    #     model = generator.generate_model()
    #     total_explored += generator.get_num_explored_models()
    #     models = [model.clip() for model in generator.collect_leaves() if not model.is_complete()]
    #     generator.close()
    return final_model, total_explored, total_time_ms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single heuristic search and log whether a model is found."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[Path("data/test.json")],
        help="Annotated story dataset(s) to load (default: data/test.json).",
    )
    parser.add_argument(
        "--story-index",
        type=int,
        default=0,
        help="Zero-based index of the story within the loaded dataset (default: 0).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="CustomJava",
        help="Registered policy name to evaluate (default: CustomJava).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=4,
        help="Worker pool size for the generator (default: 4).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=200,
        help="Timeout per generator invocation in milliseconds (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    stories = load_annotated_stories(args.dataset)
    if not stories:
        raise SystemExit("No stories loaded from the provided dataset paths.")
    if args.story_index < 0 or args.story_index >= len(stories):
        raise SystemExit(
            f"Story index {args.story_index} is out of range for {len(stories)} loaded stories."
    )
    story = stories[args.story_index]
    policy = get_policy(args.policy)
    model, explored, time_ms = _run_full_story(
        story,
        policy,
        timeout_ms=args.timeout_ms,
    )
    logging.info(
        "HeuristicStatistics[policy=%s, timeMs=%d, exploredModels=%d, modelFound=%s]",
        policy.name,
        round(time_ms),
        explored,
        model is not None,
    )
    if model is None:
        logging.warning(
            "Policy '%s' did not find a complete model for story '%s' "
            "(explored %d models, %.2f ms).",
            policy.name,
            story.synopsis,
            explored,
            time_ms,
        )
    else:
        logging.info(
            "Policy '%s' found model #%d after exploring %d models (%.2f ms).",
            policy.name,
            model.id,
            explored,
            time_ms,
        )


if __name__ == "__main__":
    main()
