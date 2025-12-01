import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from nls_python.typing_utils import typechecked

from .bank import StoryExample, StorySentence

DEFAULT_DATASET_PATHS: tuple[Path, ...] = (
    Path("data/train.json"),
    Path("data/val.json"),
    Path("data/test.json"),
)


@typechecked
def _sentence_from_annotation(sentence: Dict[str, Any], index: int) -> StorySentence:
    annotation = sentence["annotation"]
    verb = annotation["verb"]
    negated = negated = annotation["negated"]
    text = sentence["text"]
    anaphora = annotation["anaphora"]
    roles = annotation["roles"]
    semantic_types = annotation["semantic_types"]
    return StorySentence(
        verb=verb,
        roles=roles,
        negated=negated,
        text=text,
        anaphora=anaphora,
        semantic_types=semantic_types
    )


@typechecked
def load_annotated_stories(paths: Sequence[Path]) -> List[StoryExample]:
    stories: List[StoryExample] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload: List[Dict[str, Any]] = json.load(handle)
        for idx, raw in enumerate(payload):
            sentences = raw.get("sentences") or []
            if not sentences:
                continue
            story_sentences = list(
                _sentence_from_annotation(sentence, s_idx)
                for s_idx, sentence in enumerate(sentences)
                if isinstance(sentence, dict)
            )
            key = raw.get("key") or f"{path.stem}_{idx}"
            synopsis = raw.get("synopsis") or raw.get("domain") or key
            stories.append(
                StoryExample(
                    key=key,
                    synopsis=synopsis,
                    sentences=story_sentences,
                )
            )
    return stories


@typechecked
def load_story_datasets(paths: Sequence[Path] | None = None) -> List[StoryExample]:
    resolved = tuple(paths or DEFAULT_DATASET_PATHS)
    stories = load_annotated_stories(resolved)
    if not stories:
        joined = ", ".join(str(path) for path in resolved)
        raise ValueError(f"No annotated stories found in: {joined}")
    return stories


__all__ = ["load_annotated_stories", "load_story_datasets", "DEFAULT_DATASET_PATHS"]
