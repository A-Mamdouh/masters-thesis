from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import ClassVar, Generic, TypeVar

from nls_python.typing_utils import typechecked

T = TypeVar("T")


@typechecked
@total_ordering
@dataclass(frozen=True)
class Salient(Generic[T]):
    """Mirror of the Java Salient record with ordering by salience."""

    obj: T
    salience: float

    FULL: ClassVar[float] = 1.0
    DECAY_RATE: ClassVar[float] = 0.9

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Salient):
            return NotImplemented
        return self.obj == other.obj

    def __lt__(self, other: "Salient[T]") -> bool:
        if not isinstance(other, Salient):
            return NotImplemented
        return self.salience < other.salience

    def __hash__(self) -> int:
        return hash(self.obj)

    def copy(self) -> "Salient[T]":
        return Salient(self.obj, self.salience)
