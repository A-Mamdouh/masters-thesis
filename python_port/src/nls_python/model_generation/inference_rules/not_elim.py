from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from nls_python.typing_utils import typechecked

from nls_python.fol.formula import Formula
from nls_python.fol.formula.fnot import Not

from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class NotElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return False

    def apply(self, model: TableauModel) -> bool:
        applications = self._collect_applications(model)
        if not applications:
            return False
        model.add_formulas({app.output for app in applications})
        model.add_rule_applications(set(applications))
        return True

    def _collect_applications(self, model: TableauModel) -> Set["NotElimApplication"]:
        applications: Set[NotElimApplication] = set()
        seen_products: Set[Formula] = set()
        for formula in model.get_formulas():
            if not isinstance(formula, Not):
                continue
            if not isinstance(formula.inner, Not):
                continue
            inner_formula = formula.inner.inner
            if inner_formula in model.get_formulas() or inner_formula in seen_products:
                continue
            seen_products.add(inner_formula)
            application = NotElimApplication(source=formula, output=inner_formula)
            if application not in model.get_rule_applications():
                applications.add(application)
        return applications


@typechecked
@dataclass(frozen=True)
class NotElimApplication(RuleApplication):
    source: Not
    output: Formula

    def __hash__(self) -> int:
        return hash(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NotElimApplication) and self.source == other.source

    def __str__(self) -> str:
        return f"NotElim: {self.source} -> {self.output}"
