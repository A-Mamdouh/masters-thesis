from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, List, Set, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula.fand import And
from nls_python.fol.formula import Formula

from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class AndElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return False

    def apply(self, model: TableauModel) -> bool:
        applications = self._collect_applications(model)
        if not applications:
            return False
        new_formulas: Set[Formula] = set()
        for application in applications:
            new_formulas.update(application.outputs)
        model.add_formulas(new_formulas)
        model.add_rule_applications(set(applications))
        return True

    def _collect_applications(self, model: TableauModel) -> Set["AndElimApplication"]:
        applications: Set[AndElimApplication] = set()
        seen_products: Set[Formula] = set()
        formulas = model.get_formulas()
        for formula in list(formulas):
            if not isinstance(formula, And):
                continue
            outputs: List[Formula] = []
            if formula.left not in formulas and formula.left not in seen_products:
                seen_products.add(formula.left)
                outputs.append(formula.left)
            if formula.right not in formulas and formula.right not in seen_products:
                seen_products.add(formula.right)
                outputs.append(formula.right)
            if not outputs:
                continue
            application = AndElimApplication(source=formula, outputs=tuple(outputs))
            if application not in model.get_rule_applications():
                applications.add(application)
        return applications


@typechecked
@dataclass(frozen=True)
class AndElimApplication(RuleApplication):
    source: And
    outputs: Tuple[Formula, ...] = field(default_factory=tuple)

    def __hash__(self) -> int:
        return hash(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AndElimApplication) and self.source == other.source

    def output(self) -> Collection[Formula]:
        return self.outputs

    def __str__(self) -> str:
        outputs = ", ".join(str(formula) for formula in self.outputs) or "∅"
        return f"AndElim: {self.source} -> [{outputs}]"
