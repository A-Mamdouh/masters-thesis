from dataclasses import dataclass
from typing import List, Set

from nls_python.typing_utils import typechecked

from .formula import Formula
from ..term import Term, Variable
from ..substitution import Substitution


@typechecked
@dataclass(frozen=True, eq=True)
class PSymbol:
    name: str
    arity: int

    def __str__(self) -> str:
        return f"{self.name}\\arity"


@typechecked
@dataclass(frozen=True, eq=True)
class PForm(Formula):
    symbol: PSymbol
    args: List[Term]

    def __post_init__(self) -> None:
        assert len(self.args) == self.symbol.arity

    def apply_sub(self, sub: Substitution) -> "PForm":
        new_args: List[Term] = [t.apply_sub(sub) for t in self.args]
        return PForm(symbol=self.symbol, args=new_args)

    @property
    def name(self) -> str:
        return self.symbol.name

    @property
    def presidence(self) -> int:
        return 100

    def __str__(self) -> str:
        if self.symbol.arity == 0:
            return self.name
        args_str = ",".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

    def __hash__(self) -> int:
        return hash((self.symbol, tuple(self.args)))

    def get_vars(self) -> Set[Variable]:
        vars_: Set[Variable] = set()
        for arg in self.args:
            vars_.update(arg.get_vars())
        return vars_
