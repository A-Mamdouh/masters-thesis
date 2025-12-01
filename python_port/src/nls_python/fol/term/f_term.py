from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, TYPE_CHECKING

from nls_python.typing_utils import typechecked

from .term import Term
from .variable import Variable

if TYPE_CHECKING:
    from ..substitution import Substitution


@typechecked
@dataclass(frozen=True, eq=True)
class FSymbol:
    name: str
    arity: int

    def __str__(self) -> str:
        return f"{self.name}\\arity"


@typechecked
@dataclass(frozen=True, eq=True)
class FTerm(Term):
    symbol: FSymbol
    args: List[Term]

    def __post_init__(self) -> None:
        assert len(self.args) == self.symbol.arity

    def apply_sub(self, sub: Substitution) -> "FTerm":
        new_args: List[Term] = [t.apply_sub(sub) for t in self.args]
        return FTerm(symbol=self.symbol, args=new_args)

    @property
    def name(self) -> str:
        return self.symbol.name

    def get_vars(self) -> Set[Variable]:
        vars: Set[Variable] = set()
        for arg in self.args:
            vars.update(arg.get_vars())
        return set(vars)

    def __str__(self) -> str:
        if self.symbol.arity == 0:
            return self.name
        args_str = ",".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

    def __hash__(self) -> int:
        return hash((self.symbol, tuple(self.args)))
