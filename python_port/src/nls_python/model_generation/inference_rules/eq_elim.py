from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula.equals import Equals
from nls_python.fol.formula import Formula
from nls_python.fol.substitution import Substitution
from nls_python.fol.unification import unify

from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class EqElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return False

    def apply(self, model: TableauModel) -> bool:
        applications = self._collect_applications(model)
        if not applications:
            return False
        for application in applications:
            model.apply_substitution(application.substitution)
            model.add_rule_applications({application})
        return True

    def _collect_applications(self, model: TableauModel) -> Set["EqElimApplication"]:
        applications: Set[EqElimApplication] = set()
        seen_products: Set[Formula] = set()
        current_sub = model.get_substitution()
        for formula in model.get_formulas():
            if not isinstance(formula, Equals):
                continue
            maybe_sub = unify(formula.left, formula.right, model.get_substitution())
            if maybe_sub is None:
                continue
            composed = current_sub.compose(maybe_sub)
            outputs = {
                f.apply_sub(composed)
                for f in model.get_formulas()
                if f.apply_sub(composed) not in model.get_formulas()
            }
            outputs -= seen_products
            seen_products.update(outputs)
            application = EqElimApplication(formula, maybe_sub, tuple(outputs))
            if application in model.get_rule_applications():
                continue
            current_sub = composed
            applications.add(application)
        return applications


@typechecked
@dataclass(frozen=True)
class EqElimApplication(RuleApplication):
    source: Equals
    substitution: Substitution
    outputs: Tuple[Formula, ...] = field(default_factory=tuple)

    def __hash__(self) -> int:
        return hash(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EqElimApplication) and self.source == other.source

    def __str__(self) -> str:
        outputs = ", ".join(str(formula) for formula in self.outputs) or "∅"
        return f"EqElim: {self.source} using {self.substitution} -> [{outputs}]"
