from dataclasses import dataclass
from typing import Set

from nls_python.typing_utils import typechecked

from ..term import Term, Variable
from .formula import Formula


@typechecked
@dataclass(frozen=True, eq=True)
class Equals(Formula):
    left: Term
    right: Term

    def apply_sub(self, sub):
        return Equals(left=self.left.apply_sub(sub), right=self.right.apply_sub(sub))

    def __str__(self) -> str:
        return f"{self.left} = {self.right}"

    @property
    def presidence(self) -> int:
        return 100

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def get_vars(self) -> Set[Variable]:
        vars_: Set[Variable] = set()
        vars_.update(self.left.get_vars())
        vars_.update(self.right.get_vars())
        return vars_
