from __future__ import annotations

from dataclasses import dataclass
from typing import Set, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from .term import Term
from .variable import Variable

if TYPE_CHECKING:
    from ..substitution import Substitution


@typechecked
@dataclass(frozen=True, eq=True)
class Constant(Term):
    name: str

    def apply_sub(self, sub: "Substitution") -> Term:
        return self

    def get_vars(self) -> Set["Variable"]:
        return set()

    def __str__(self) -> str:
        return self.name
