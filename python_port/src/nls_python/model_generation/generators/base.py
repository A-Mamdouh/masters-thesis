
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, List, Optional, Sequence, Tuple

from nls_python.typing_utils import typechecked

from ..consistency import ConsistencyCheck, default_consistency_checks
from ..heuristics import ModelCostFunction, basic_model_features, depth_biased_cost
from ..inference_rules import (
    AndElim,
    EqElim,
    ExistElim,
    ForallElim,
    InferenceRule,
    NotElim,
    OrElim,
)
from ..tableau_model import TableauModel


@dataclass
class WorkerResult:
    index: int
    results: List[TableauModel]


@typechecked
class BaseModelGenerator(ABC):
    """Shared scaffolding for the different generator strategies."""

    def __init__(
        self,
        initial_model: TableauModel,
        *,
        inference_rules: Sequence[InferenceRule] | None = None,
        consistency_checks: Iterable[ConsistencyCheck] | None = None,
        cost_function: ModelCostFunction | None = None,
    ) -> None:
        self.initial_model = initial_model.clip()
        rules = list(inference_rules or self.default_inference_rules())
        self._rules_branching = tuple(rule for rule in rules if rule.branching)
        self._rules_non_branching = tuple(rule for rule in rules if not rule.branching)
        checks = set(default_consistency_checks())
        if consistency_checks is not None:
            checks.update(consistency_checks)
        self._consistency_checks = tuple(checks)
        self._cost_function = cost_function or depth_biased_cost
        self.pool_size = 1
        self.max_results = 1
        self._executor = ThreadPoolExecutor(max_workers=self.pool_size, thread_name_prefix="tableau")
        self._results: List[TableauModel] = []
        self._timeout_ms = 0
        self._num_explored = 0
        self._lock = Lock()
        self._branch_events = 0
        self._branch_models = 0

    @staticmethod
    def default_inference_rules() -> List[InferenceRule]:
        return [AndElim(), ExistElim(), ForallElim(), NotElim(), OrElim(), EqElim()]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def set_timeout(self, timeout_ms: int) -> None:
        self._timeout_ms = max(0, timeout_ms)

    def get_timeout(self) -> int:
        return self._timeout_ms

    def clear_top_result(self) -> None:
        if self._results:
            self._results.pop(0)

    def get_num_explored_models(self) -> int:
        return self._num_explored

    def _copy_checks(self) -> Tuple[ConsistencyCheck, ...]:
        return tuple(check.copy() for check in self._consistency_checks)

    def _record_results(self, *models: TableauModel) -> None:
        with self._lock:
            for model in models:
                self._results.append(model.clip())

    def _increment_explored(self, amount: int) -> None:
        with self._lock:
            self._num_explored += amount

    def get_branch_statistics(self) -> Tuple[int, int]:
        return self._branch_events, self._branch_models

    def _score_model(self, model: TableauModel) -> float:
        features = basic_model_features(model)
        return self._cost_function(model, features)
    
    def _expand_sub_model(self, model: TableauModel, *, index: int = 0) -> Tuple[Optional[TableauModel], List[TableauModel]]:
        if model.is_complete():
            self._increment_explored(1)
            return model, []
        model_changed = False
        done = False
        while not done:
            done = True
            for rule in self._rules_non_branching:
                if rule.apply(model):
                    model_changed = True
                    done = False
        branched = False
        for rule in self._rules_branching:
            if rule.apply(model):
                branched = True
                break
        if not model.check_consistency(self._copy_checks()):
            self._increment_explored(1)
            return None, []
        new_frontier: List[TableauModel] = []
        if branched:
            for branch_set in model.get_extensions().values():
                new_frontier.extend(filter(lambda model: model.check_consistency(self._copy_checks()), branch_set))
            self._branch_events += 1
            self._branch_models += len(new_frontier)
        else:
            if not model_changed:
                model.complete()
            new_frontier.append(model)
        return None, new_frontier

    def _expand_model(self, model: TableauModel, *, index: int = 0) -> WorkerResult:
        worker_frontier: List[TableauModel] = [model]
        results: List[TableauModel] = []
        while worker_frontier:
            model, *worker_frontier = worker_frontier
            result, frontier = self._expand_sub_model(model, index=index)
            if result is not None:
                results.append(result)
            worker_frontier.extend(frontier)
        return WorkerResult(index=index, results=results)



    @abstractmethod
    def generate_model(self, *, verbose: bool = False) -> TableauModel | None: ...
