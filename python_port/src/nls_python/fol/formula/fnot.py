from dataclasses import dataclass
from typing import Set

from nls_python.typing_utils import typechecked

from .formula import Formula
from ..substitution import Substitution
from ..term import Variable


@typechecked
@dataclass(frozen=True, eq=True)
class Not(Formula):
    inner: Formula

    def apply_sub(self, sub: Substitution) -> "Not":
        return Not(inner=self.inner.apply_sub(sub))

    def __str__(self) -> str:
        if self.inner.presidence < self.presidence:
            return f"-({self.inner})"
        return f"-{self.inner}"

    @property
    def presidence(self) -> int:
        return 90

    def get_vars(self) -> Set[Variable]:
        return set(self.inner.get_vars())
