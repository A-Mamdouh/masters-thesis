from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from nls_python.typing_utils import typechecked

from .rankers import StoryOrderer
from ..tableau_model import TableauModel

try:  # pragma: no cover
    import ollama
except ImportError:  # pragma: no cover
    ollama = None  # type: ignore[assignment]


@typechecked
def _model_description(model: TableauModel) -> str:
    narrative_text = model.get_narrative().story_text()
    if narrative_text:
        return narrative_text
    return model.create_string()


@typechecked
def _normalize_indices(order: Iterable[int], total: int) -> List[int]:
    normalized: List[int] = []
    seen = set()
    for raw in order:
        if raw < 0 or raw >= total:
            continue
        if raw in seen:
            continue
        normalized.append(raw)
        seen.add(raw)
    for idx in range(total):
        if idx not in seen:
            normalized.append(idx)
    return normalized


@typechecked
def normal_order_models(
    orderer: StoryOrderer, models: Sequence[TableauModel]
) -> List[TableauModel]:
    if not models:
        return []
    story_text = models[0].get_narrative().story_text()
    descriptions = [_model_description(model) for model in models]
    indices = _normalize_indices(orderer.order(story_text, descriptions), len(models))
    return [models[idx] for idx in indices]


class _OrderForest:
    """Caches pairwise relations to avoid repeated ranker calls."""

    def __init__(self) -> None:
        self._graph: dict[int, set[int]] = {}

    def is_smaller(self, first: int, second: int) -> bool | None:
        if self._reachable(first, second):
            self._record(first, second)
            return True
        if self._reachable(second, first):
            self._record(second, first)
            return False
        return None

    def _reachable(self, source: int, target: int) -> bool:
        frontier = list(self._graph.get(source, set()))
        visited = set()
        while frontier:
            node = frontier.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            frontier.extend(self._graph.get(node, set()))
        return False

    def _record(self, smaller: int, larger: int) -> None:
        self._graph.setdefault(smaller, set()).add(larger)

    def set_relation(self, smaller: int, larger: int, is_smaller: bool) -> None:
        if is_smaller:
            self._record(smaller, larger)
        else:
            self._record(larger, smaller)


@dataclass
class _ComparableNarrativeModel:
    orderer: StoryOrderer
    cache: _OrderForest
    story_text: str
    description: str
    model: TableauModel
    key: int

    def __lt__(self, other: "_ComparableNarrativeModel") -> bool:
        cached = self.cache.is_smaller(self.key, other.key)
        if cached is not None:
            return cached
        indices = self.orderer.order(
            self.story_text,
            [other.description, self.description],
        )
        normalized = _normalize_indices(indices, 2)
        is_smaller = normalized[0] == 1
        self.cache.set_relation(self.key, other.key, is_smaller)
        return is_smaller


@typechecked
def pairwise_order_models(
    orderer: StoryOrderer, models: Sequence[TableauModel]
) -> List[TableauModel]:
    if not models:
        return []
    story_text = models[0].get_narrative().story_text()
    cache = _OrderForest()
    comparables: List[_ComparableNarrativeModel] = []
    for idx, model in enumerate(models):
        comparable = _ComparableNarrativeModel(
            orderer=orderer,
            cache=cache,
            story_text=story_text,
            description=_model_description(model),
            model=model,
            key=idx,
        )
        if not comparables:
            comparables.append(comparable)
            continue
        inserted = False
        for pos, existing in enumerate(comparables):
            if comparable < existing:
                comparables.insert(pos, comparable)
                inserted = True
                break
        if not inserted:
            comparables.append(comparable)
    return [comparable.model for comparable in comparables]


@typechecked
def unique_models(models: Iterable[TableauModel]) -> List[TableauModel]:
    seen = set()
    unique: List[TableauModel] = []
    for model in models:
        signature = model.get_narrative().story_text() or model.create_string()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(model)
    return unique


@typechecked
def order_story_models(
    orderer: StoryOrderer,
    models: Sequence[TableauModel],
    *,
    pair_threshold: int = 4,
) -> List[TableauModel]:
    deduped = unique_models(models)
    if len(deduped) > pair_threshold:
        return pairwise_order_models(orderer, deduped)
    return normal_order_models(orderer, deduped)


@dataclass
class OllamaStoryOrderer:
    """Thin wrapper around `ollama.generate` for reproducible ordering prompts."""

    model_name: str = "llama3.1"
    temperature: float = 0.0
    seed: int = 42

    def order(
        self, story_text: str, model_descriptions: Sequence[str]
    ) -> Sequence[int]:
        if ollama is None:  # pragma: no cover
            raise RuntimeError(
                "ollama is not installed. Install the `ollama` package to enable this ranker."
            )
        if not model_descriptions:
            return ()
        models_str = "\n".join(
            f"{idx + 1}. {desc}" for idx, desc in enumerate(model_descriptions)
        )
        prompt = (
            f'Given the short story "{story_text}", order the interpretations '
            f"from most to least reasonable using their numbers:\n{models_str}"
        )
        system = (
            'Respond with JSON {"order": [Number], "reasoning": "string"} '
            f"covering all {len(model_descriptions)} interpretations."
        )
        response = ollama.generate(  # pragma: no cover
            model=self.model_name,
            system=system,
            prompt=prompt,
            format="json",
            options={"temperature": self.temperature, "seed": self.seed},
        )
        payload = response["response"]
        try:
            import json
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("json module unavailable") from exc
        parsed = json.loads(payload)
        indices = []
        for entry in parsed.get("order", []):
            if isinstance(entry, int):
                indices.append(entry - 1)
        return _normalize_indices(indices, len(model_descriptions))


@dataclass
class LexicographicStoryOrderer:
    """Deterministic fallback orderer that sorts by the textual description."""

    def order(
        self, story_text: str, model_descriptions: Sequence[str]
    ) -> Sequence[int]:
        indices = sorted(
            range(len(model_descriptions)), key=lambda idx: model_descriptions[idx]
        )
        return indices


__all__ = [
    "normal_order_models",
    "pairwise_order_models",
    "unique_models",
    "order_story_models",
    "OllamaStoryOrderer",
    "LexicographicStoryOrderer",
]
