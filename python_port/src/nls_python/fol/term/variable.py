from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import ClassVar, Set, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from .term import Term

if TYPE_CHECKING:
    from ..substitution import Substitution


@typechecked
@dataclass(frozen=True, eq=True)
class Variable(Term):
    name: str
    _num: ClassVar[int] = 0
    _lock: ClassVar[Lock] = Lock()

    @staticmethod
    def make() -> "Variable":
        with Variable._lock:
            Variable._num += 1
            current = Variable._num
        name: str = f"_V{current}"
        return Variable(name)

    def get_vars(self) -> Set["Variable"]:
        return {self}

    def apply_sub(self, sub: "Substitution") -> Term:
        return sub.get(self, self)

    def __str__(self) -> str:
        return self.name
