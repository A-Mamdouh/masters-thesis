from dataclasses import dataclass, field
from typing import Any, List, Set
from nls_python.typing_utils import typechecked

from .formula import Formula
from ..term import Term, SkolemFunction, Variable
from ..substitution import Substitution
from ..utils import is_literal


@typechecked
@dataclass(frozen=True, eq=True)
class Exists(Formula):
    var: Variable
    precondition: Formula
    inner: Formula
    _skolem_references: List[Term] = field(default_factory=list)

    def apply_sub(self, sub: Substitution) -> "Exists":
        sub = sub.without(self.var)
        return Exists(
            self.var,
            precondition=self.precondition.apply_sub(sub),
            inner=self.inner.apply_sub(sub),
        )

    def skolemize(self, var: Variable, term: Term) -> "Exists":
        new_skolem_references = [*self._skolem_references, term]
        sub = Substitution()
        sub.put(var, term)
        return Exists(
            var=self.var,
            precondition=self.precondition.apply_sub(sub),
            inner=self.inner.apply_sub(sub),
            _skolem_references=new_skolem_references,
        )

    def get_witness(self) -> Term:
        if len(self._skolem_references) == 0:
            return SkolemFunction([])
        return SkolemFunction(self._skolem_references)

    def apply(self, term: Term) -> Formula:
        if isinstance(term, SkolemFunction):
            assert (
                len(set(self._skolem_references).symmetric_difference(set(term.args)))
                == 0
            )
        sub: Substitution = Substitution()
        sub.put(self.var, term)
        return self.inner.apply_sub(sub)

    def apply_precondition(self, term: Term) -> Formula:
        if isinstance(term, SkolemFunction):
            assert (
                len(set(self._skolem_references).symmetric_difference(set(term.args)))
                == 0
            )
        sub: Substitution = Substitution()
        sub.put(self.var, term)
        return self.precondition.apply_sub(sub)

    def __str__(self) -> str:
        if self.inner.presidence < self.presidence:
            return f"∃{self.var}:{self.precondition}(.{self.inner})"
        return f"∃{self.var}:{self.precondition}.{self.inner}"

    @property
    def presidence(self) -> int:
        if is_literal(self.inner):
            return self.inner.presidence
        return 70

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Exists):
            return False
        if (
            len(
                set(other._skolem_references).symmetric_difference(
                    set(self._skolem_references)
                )
            )
            != 0
        ):
            return False
        witness = self.get_witness()
        return self.apply(witness) == other.apply(witness) and self.apply_precondition(
            witness
        ) == other.apply_precondition(witness)

    def __hash__(self) -> int:
        placeholder = Variable("__exists_placeholder")
        sub = Substitution()
        sub.put(self.var, placeholder)
        normalized_inner = self.inner.apply_sub(sub)
        normalized_precondition = self.precondition.apply_sub(sub)
        references = frozenset(self._skolem_references)
        return hash((normalized_precondition, normalized_inner, references))

    def get_vars(self) -> Set[Variable]:
        vars_: Set[Variable] = set()
        vars_.update(self.inner.get_vars())
        vars_.update(self.precondition.get_vars())
        vars_.discard(self.var)
        return vars_

    def free_vars(self) -> Set[Variable]:
        return self.get_vars()
