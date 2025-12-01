from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable, Protocol

from nls_python.typing_utils import typechecked

from .tableau_model import TableauModel


@dataclass(frozen=True)
class ModelFeatures:
    """Light-weight summary of a tableau model used for heuristics."""

    model_id: int
    num_formulas: int
    num_individuals: int
    num_extensions: int
    num_pending_branches: int
    num_rule_applications: int
    sentence_depth: int
    complete: bool


def basic_model_features(model: TableauModel) -> ModelFeatures:
    extensions = model.get_extensions()
    pending_branches = sum(len(branches) for branches in extensions.values())
    return ModelFeatures(
        model_id=model.id,
        num_formulas=len(model.get_formulas()),
        num_individuals=len(model.get_individuals()),
        num_extensions=len(extensions),
        num_pending_branches=pending_branches,
        num_rule_applications=len(model.get_rule_applications()),
        sentence_depth=model.get_sentence_depth(),
        complete=model.is_complete(),
    )


class ModelCostFunction(Protocol):
    """Protocol implemented by heuristic / policy evaluators."""

    def __call__(self, model: TableauModel, features: ModelFeatures) -> float: ...


@typechecked
class StaticCostFunction:
    """Simple adapter around a callable that consumes only the features."""

    def __init__(self, scorer: Callable[[ModelFeatures], float]) -> None:
        self._scorer = scorer

    def __call__(self, model: TableauModel, features: ModelFeatures) -> float:
        return float(self._scorer(features))


@typechecked
class NeuralPolicyAdapter:
    """Adapter that allows calling out to a neural policy function later on."""

    def __init__(
        self,
        predictor: Callable[..., float],
        *,
        feature_preprocessor: Callable[[ModelFeatures], ModelFeatures] | None = None,
    ) -> None:
        self._predictor = predictor
        self._preprocessor = feature_preprocessor
        self._mode = self._determine_mode(predictor)

    def __call__(self, model: TableauModel, features: ModelFeatures) -> float:
        processed = self._preprocessor(features) if self._preprocessor else features
        if self._mode == "features":
            return float(self._predictor(processed))
        return float(self._predictor(model, processed))

    @staticmethod
    def _determine_mode(predictor: Callable[..., float]) -> str:
        params = signature(predictor).parameters
        if len(params) == 1:
            return "features"
        if len(params) == 2:
            return "model"
        raise TypeError(
            "Neural policy predictors must accept either (features) or (model, features)."
        )


def depth_biased_cost(_model: TableauModel, features: ModelFeatures) -> float:
    """A default heuristic that favours deeper, smaller branches."""
    branch_penalty = features.num_pending_branches or 1
    return (features.sentence_depth + 1) * branch_penalty + features.num_formulas * 0.1


def bfs_cost(_model: TableauModel, features: ModelFeatures) -> float:
    """Breadth-first style ordering using the tableau identifier."""
    return float(features.model_id)


def dfs_cost(_model: TableauModel, features: ModelFeatures) -> float:
    """Depth-first style ordering favouring recently created tableaux."""
    return float(-features.model_id)


_JAVA_DEPTH_WEIGHT = 1_000_000
_JAVA_INDIVIDUAL_WEIGHT = 1_000


def java_custom_cost(_model: TableauModel, features: ModelFeatures) -> float:
    """Replica of the Java custom comparator (depth, individuals, formulas desc)."""
    return (
        features.sentence_depth * _JAVA_DEPTH_WEIGHT
        + features.num_individuals * _JAVA_INDIVIDUAL_WEIGHT
        - features.num_formulas
    )


__all__ = [
    "ModelFeatures",
    "basic_model_features",
    "ModelCostFunction",
    "StaticCostFunction",
    "NeuralPolicyAdapter",
    "depth_biased_cost",
    "bfs_cost",
    "dfs_cost",
    "java_custom_cost",
]
