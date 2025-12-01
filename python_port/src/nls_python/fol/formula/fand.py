from dataclasses import dataclass
from typing import Set

from nls_python.typing_utils import typechecked

from .formula import Formula
from ..substitution import Substitution
from ..term import Variable


@typechecked
@dataclass(frozen=True, eq=True)
class And(Formula):
    left: Formula
    right: Formula

    def apply_sub(self, sub: Substitution) -> "And":
        return And(left=self.left.apply_sub(sub), right=self.right.apply_sub(sub))

    def __str__(self) -> str:
        left: str = f"{self.left}"
        if self.left.presidence < self.presidence:
            left = f"({left})"
        right: str = f"{self.right}"
        if self.right.presidence < self.presidence:
            right = f"({right})"
        return f"{left} & {right}"

    @property
    def presidence(self) -> int:
        return 80

    def get_vars(self) -> Set[Variable]:
        vars_: Set[Variable] = set()
        vars_.update(self.left.get_vars())
        vars_.update(self.right.get_vars())
        return vars_
