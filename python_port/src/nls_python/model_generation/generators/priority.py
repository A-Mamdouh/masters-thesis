from __future__ import annotations

import heapq
import time
from itertools import count
from typing import List, Sequence

from .base import BaseModelGenerator, WorkerResult
from ..tableau_model import TableauModel


class PriorityModelGenerator(BaseModelGenerator):
    """A faithful port of the Java priority-queue model generator."""

    def __init__(
        self,
        *args,
        initial_models: Sequence[TableauModel] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frontier: List[tuple[float, int, TableauModel]] = []
        self._counter = count()
        self._drain_size = 1
        self._enqueue_model(self.initial_model)
        if initial_models:
            self.add_initial_models(initial_models)

    def _enqueue_model(self, model: TableauModel) -> None:
        heapq.heappush(
            self._frontier,
            (self._score_model(model), next(self._counter), model.clip()),
        )

    def _drain_batch(self) -> List[TableauModel]:
        batch: List[TableauModel] = []
        drain = min(self._drain_size, len(self._frontier))
        for _ in range(drain):
            _, _, model = heapq.heappop(self._frontier)
            batch.append(model)
        return batch

    def _handle_worker_result(self, worker_result: WorkerResult) -> None:
        if worker_result.results:
            self._record_results(*worker_result.results)

    def add_initial_models(self, models: Sequence[TableauModel]) -> None:
        for model in models:
            self._enqueue_model(model.clip())

    def generate_model(self, *, verbose: bool = False) -> TableauModel | None:
        start = time.monotonic()
        timed_out = False
        while len(self._results) < self.max_results and self._frontier:
            if (
                self._timeout_ms
                and (time.monotonic() - start) * 1000 > self._timeout_ms
            ):
                timed_out = True
                break
            batch = self._drain_batch()
            if not batch:
                break
            futures = [
                self._executor.submit(self._expand_model, model, index=index)
                for index, model in enumerate(batch)
            ]
            for future in futures:
                worker_result = future.result()
                self._handle_worker_result(worker_result)
        if verbose:
            print(f"Timed out: {timed_out}")
        if self._results:
            return self._results[0].clip()
        for _, _, model in self._frontier:
            if model.check_consistency(self._copy_checks()):
                return model.clip()
        return None

    def collect_leaves(self) -> List[TableauModel]:
        leaves = [result.clip() for result in self._results]
        leaves.extend(model.clip() for _, _, model in self._frontier)
        return leaves
