from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula.exists import Exists
from nls_python.fol.formula.fnot import Not
from nls_python.fol.formula import Formula
from nls_python.fol.substitution import Substitution
from nls_python.fol.term import Term

from ..salient import Salient
from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class ForallElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return False

    def apply(self, model: TableauModel) -> bool:
        applications = self._collect_applications(model)
        if not applications:
            return False
        for application in applications:
            model.add_formulas({formula for _, formula in application.outputs})
        model.add_rule_applications(set(applications))
        return True

    def _collect_applications(
        self, model: TableauModel
    ) -> Set["ForallElimApplication"]:
        applications: Set[ForallElimApplication] = set()
        seen_products: Set[Formula] = set()
        for formula in model.get_formulas():
            match = self._match_forall(formula)
            if match is None:
                continue
            exists_formula, body = match
            outputs: Dict[Term, Formula] = {}
            for individual in model.get_individuals():
                precondition = exists_formula.apply_precondition(individual.obj)
                if precondition not in model.get_formulas():
                    continue
                instantiated = self._instantiate_body(
                    exists_formula, body, individual.obj
                )
                if (
                    instantiated in model.get_formulas()
                    or instantiated in seen_products
                ):
                    continue
                outputs[individual.obj] = instantiated
                seen_products.add(instantiated)
                model.set_individuals_salience({Salient(individual.obj, Salient.FULL)})
            if not outputs:
                continue
            assert isinstance(formula, Not)
            application = ForallElimApplication(
                source=formula, outputs=tuple(outputs.items())
            )
            if application not in model.get_rule_applications():
                applications.add(application)
        return applications

    def _match_forall(self, formula: Formula) -> Optional[Tuple[Exists, Formula]]:
        if not isinstance(formula, Not):
            return None
        exists_formula = formula.inner
        if not isinstance(exists_formula, Exists):
            return None
        body = exists_formula.inner
        if isinstance(body, Not):
            body = body.inner
        else:
            body = Not(body)
        return exists_formula, body

    def _instantiate_body(
        self, exists_formula: Exists, body: Formula, term: Term
    ) -> Formula:
        substitution = Substitution()
        substitution.put(exists_formula.var, term)
        return body.apply_sub(substitution)


@typechecked
@dataclass(frozen=True)
class ForallElimApplication(RuleApplication):
    source: Not
    outputs: Tuple[Tuple[Term, Formula], ...]

    def __hash__(self) -> int:
        return hash((self.source, tuple(term for term, _ in self.outputs)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForallElimApplication):
            return False
        return self.source == other.source and {term for term, _ in self.outputs} == {
            term for term, _ in other.outputs
        }

    def __str__(self) -> str:
        if not self.outputs:
            return f"ForallElim: {self.source}"
        mapping = ", ".join(f"{term}->{formula}" for term, formula in self.outputs)
        return f"ForallElim: {self.source} -> [{mapping}]"
