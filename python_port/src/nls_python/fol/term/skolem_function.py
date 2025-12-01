from __future__ import annotations

from threading import Lock
from dataclasses import dataclass, field
from typing import ClassVar, List, Set, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from .f_term import FSymbol
from .term import Term

if TYPE_CHECKING:
    from .variable import Variable
    from ..substitution import Substitution


@typechecked
@dataclass(frozen=True, eq=True)
class SkolemFunction(Term):
    args: List[Term]
    symbol: FSymbol = field(init=False)
    _num: ClassVar[int] = 0
    _lock: ClassVar[Lock] = Lock()

    def __post_init__(self) -> None:
        with SkolemFunction._lock:
            SkolemFunction._num += 1
            current = SkolemFunction._num
        symbol: FSymbol = FSymbol(f"SK_{current}", len(self.args))
        object.__setattr__(self, "symbol", symbol)

    def apply_sub(self, sub: Substitution) -> "Term":
        new_args: List[Term] = [arg.apply_sub(sub) for arg in self.args]
        term: SkolemFunction = SkolemFunction(new_args)
        object.__setattr__(term, "symbol", self.symbol)
        return sub.get(term, term)

    def get_vars(self) -> Set[Variable]:
        vars: Set[Variable] = set()
        for arg in self.args:
            vars.update(arg.get_vars())
        return set(vars)

    def __str__(self) -> str:
        if self.symbol.arity == 0:
            return self.symbol.name
        args_str = ",".join(str(arg) for arg in self.args)
        return f"{self.symbol.name}({args_str})"

    def __hash__(self) -> int:
        return hash((self.symbol, tuple(self.args)))
