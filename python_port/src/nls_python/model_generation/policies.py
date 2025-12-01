from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from nls_python.typing_utils import typechecked

from .heuristics import (
    ModelCostFunction,
    ModelFeatures,
    NeuralPolicyAdapter,
    StaticCostFunction,
    basic_model_features,
    bfs_cost,
    depth_biased_cost,
    dfs_cost,
    java_custom_cost,
)
from .tableau_model import TableauModel


@typechecked
@dataclass(frozen=True)
class SearchPolicy:
    """Metadata wrapper describing how to order tableau models."""

    name: str
    cost_function: ModelCostFunction
    drain_size: int

    def score(self, model: TableauModel) -> float:
        return self.cost_function(model, basic_model_features(model))


def _make_policy(name: str, cost: ModelCostFunction, drain_size: int) -> SearchPolicy:
    return SearchPolicy(name=name, cost_function=cost, drain_size=drain_size)


BFS_POLICY = _make_policy("BFS", bfs_cost, drain_size=-1)
DFS_POLICY = _make_policy("DFS", dfs_cost, drain_size=1)
JAVA_CUSTOM_POLICY = _make_policy("CustomJava", java_custom_cost, drain_size=12)
DEPTH_POLICY = _make_policy("DepthBiased", depth_biased_cost, drain_size=-1)


_POLICY_REGISTRY: Dict[str, SearchPolicy] = {
    policy.name: policy
    for policy in (BFS_POLICY, DFS_POLICY, JAVA_CUSTOM_POLICY, DEPTH_POLICY)
}


def register_policy(policy: SearchPolicy, *, override: bool = False) -> None:
    if not override and policy.name in _POLICY_REGISTRY:
        raise ValueError(f"Policy '{policy.name}' already registered")
    _POLICY_REGISTRY[policy.name] = policy


def list_policies() -> List[SearchPolicy]:
    return list(_POLICY_REGISTRY.values())


def get_policy(name: str) -> SearchPolicy:
    try:
        return _POLICY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy '{name}'. Available: {list(_POLICY_REGISTRY)}"
        ) from exc


def make_static_policy(
    name: str,
    scorer: Callable[[ModelFeatures], float],
    *,
    drain_size: int = -1,
) -> SearchPolicy:
    return _make_policy(name, StaticCostFunction(scorer), drain_size=drain_size)


def make_neural_policy(
    name: str,
    predictor: Callable[[ModelFeatures], float],
    *,
    drain_size: int = -1,
    feature_preprocessor: Callable[[ModelFeatures], ModelFeatures] | None = None,
) -> SearchPolicy:
    adapter = NeuralPolicyAdapter(predictor, feature_preprocessor=feature_preprocessor)
    return _make_policy(name, adapter, drain_size=drain_size)


__all__ = [
    "SearchPolicy",
    "BFS_POLICY",
    "DFS_POLICY",
    "JAVA_CUSTOM_POLICY",
    "DEPTH_POLICY",
    "register_policy",
    "list_policies",
    "get_policy",
    "make_static_policy",
    "make_neural_policy",
]
