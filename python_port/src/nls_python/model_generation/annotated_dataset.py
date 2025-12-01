#!/usr/bin/env python3
"""Utilities for converting annotated story JSON into literal-centric datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from nls_python.typing_utils import typechecked


def _load_stories(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of stories in {path}, got {type(payload)!r}")
    return payload


def _resolve_roles(annotation: Mapping[str, object]) -> List[Tuple[str, str]]:
    """Return deterministic (role, value) pairs resolved through anaphora."""
    roles = annotation.get("roles") or {}
    anaphora = annotation.get("anaphora") or {}
    resolved: List[Tuple[str, str]] = []
    if isinstance(roles, Mapping):
        for role, value in roles.items():
            if not isinstance(role, str) or not isinstance(value, str):
                continue
            referent = anaphora.get(value, value)
            resolved.append((role, str(referent)))
    if not resolved:
        arguments = annotation.get("arguments")
        if isinstance(arguments, list):
            resolved = [(f"arg{idx}", str(arg)) for idx, arg in enumerate(arguments)]
    return sorted(resolved, key=lambda item: item[0])


def _literal_from_annotation(annotation: Mapping[str, object]) -> Tuple[str, Dict[str, str]]:
    verb = annotation.get("verb")
    if not isinstance(verb, str):
        raise ValueError("Annotation must include 'verb'.")
    resolved_roles = dict(_resolve_roles(annotation))
    arguments = list(resolved_roles.values())
    literal_inner = ", ".join(arguments)
    literal = f"{verb}({literal_inner})" if literal_inner else f"{verb}()"
    if bool(annotation.get("negated")):
        literal = f"NOT {literal}"
    return literal, resolved_roles


def _resolve_text(text: str, anaphora: Mapping[str, str]) -> str:
    resolved = text
    for pronoun, referent in anaphora.items():
        if not pronoun or not referent:
            continue
        pattern = re.compile(rf"\b{re.escape(pronoun)}\b")
        resolved = pattern.sub(referent, resolved)
    return resolved


def _story_samples(
    story: Mapping[str, object],
    *,
    story_id: str,
    source: Path,
    strict: bool,
) -> Tuple[List[Dict[str, object]], int]:
    sentences = story.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"Story '{story_id}' has no 'sentences' array.")
    synopsis = str(story.get("synopsis", ""))
    domain = str(story.get("domain", "unknown"))
    context_texts: List[str] = []
    context_literals: List[str] = []
    samples: List[Dict[str, object]] = []
    skipped = 0
    for index, raw_sentence in enumerate(sentences):
        if not isinstance(raw_sentence, Mapping):
            raise ValueError(f"Sentence #{index} in '{story_id}' is not an object.")
        annotation = raw_sentence.get("annotation")
        if not isinstance(annotation, Mapping):
            if strict:
                raise ValueError(f"Sentence #{index} in '{story_id}' lacks annotation.")
            skipped += 1
            continue
        literal, resolved_roles = _literal_from_annotation(annotation)
        anaphora = annotation.get("anaphora", {})
        if not isinstance(anaphora, Mapping):
            anaphora = {}
        text = str(raw_sentence.get("text", ""))
        resolved_text = _resolve_text(text, anaphora) if anaphora else text
        sample = {
            "source_file": str(source),
            "story_id": story_id,
            "story_domain": domain,
            "story_synopsis": synopsis,
            "story_sentence_count": len(sentences),
            "sentence_index": index,
            "prefix_length": index,
            "context_text": list(context_texts),
            "context_literals": list(context_literals),
            "text": text,
            "resolved_text": resolved_text,
            "verb": annotation.get("verb"),
            "negated": bool(annotation.get("negated", False)),
            "literal": literal,
            "roles": annotation.get("roles", {}),
            "resolved_roles": resolved_roles,
            "semantic_types": annotation.get("semantic_types", {}),
            "anaphora": dict(anaphora),
        }
        samples.append(sample)
        context_texts.append(text)
        context_literals.append(literal)
    return samples, skipped


@typechecked
def build_literal_dataset(
    inputs: Sequence[Path],
    output_path: Path,
    *,
    strict: bool = False,
) -> Tuple[int, int]:
    """Flatten annotated stories into JSONL entries with resolved literals."""
    samples: List[Dict[str, object]] = []
    skipped = 0
    for path in inputs:
        stories = _load_stories(path)
        prefix = path.stem
        for idx, story in enumerate(stories):
            story_id = f"{prefix}_{idx:03d}"
            story_samples, story_skipped = _story_samples(
                story, story_id=story_id, source=path, strict=strict
            )
            samples.extend(story_samples)
            skipped += story_skipped
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    return len(samples), skipped


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annotated story JSON into literal-level JSONL samples.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input story JSON files (e.g., data/train.json data/val.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise if any sentence lacks annotation (default: skip silently).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    count, skipped = build_literal_dataset(
        args.inputs,
        args.output,
        strict=args.strict,
    )
    extra = f" (skipped {skipped} sentences without annotations)" if skipped else ""
    print(f"Wrote {count} literal records to {args.output}{extra}")


if __name__ == "__main__":
    main()
