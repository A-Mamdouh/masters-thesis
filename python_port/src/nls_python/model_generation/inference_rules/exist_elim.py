from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula.exists import Exists
from nls_python.fol.term import Term, SkolemFunction

from ..salient import Salient
from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class ExistElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return True

    def apply(self, model: TableauModel) -> bool:
        exists_formula = self._find_exist(model)
        if exists_formula is None:
            return False
        application = self._build_application(exists_formula, model)
        if application is None:
            return False
        extensions: Dict[RuleApplication, Set[TableauModel]] = {
            application: {branch for _, branch in application.output_extensions}
        }
        model.add_extensions(extensions)
        model.add_rule_applications({application})
        return True

    def _find_exist(self, model: TableauModel) -> Optional[Exists]:
        for formula in model.get_formulas():
            if isinstance(formula, Exists):
                empty = ExistElimApplication(formula, (), tuple())
                if empty not in model.get_rule_applications():
                    return formula
        return None

    def _build_application(
        self, exists: Exists, model: TableauModel
    ) -> Optional["ExistElimApplication"]:
        branches: Dict[Term, TableauModel] = {}
        individuals = sorted(model.get_individuals(), reverse=True)
        for individual in individuals:
            precondition = exists.apply_precondition(individual.obj)
            if precondition not in model.get_formulas():
                continue
            instantiated = exists.apply(individual.obj)
            branch_model = model.branch_copy()
            branch_model.add_formulas({instantiated})
            branch_model.add_rule_applications({ExistElimApplication(exists, (individual.obj,), tuple())})
            branch_model.set_individuals_salience(
                {Salient(individual.obj, Salient.FULL)}
            )
            branches[individual.obj] = branch_model

        witness_vars = sorted(exists.free_vars(), key=lambda var: var.name)
        witness = SkolemFunction(list(witness_vars))
        witness_formula = exists.apply(witness)
        witness_precondition = exists.apply_precondition(witness)
        witness_model = model.branch_copy()
        witness_model.add_formulas({witness_formula, witness_precondition})
        witness_model.add_individuals({witness}, Salient.FULL)
        witness_model.add_rule_applications({ExistElimApplication(exists, (witness,), tuple())})
        branches[witness] = witness_model

        if not branches:
            return None
        outputs = tuple(branches.items())
        return ExistElimApplication(exists, tuple(branches.keys()), outputs)


@typechecked
@dataclass(frozen=True)
class ExistElimApplication(RuleApplication):
    source: Exists
    initializers: Tuple[Term, ...] = field(compare=False)
    output_extensions: Tuple[Tuple[Term, TableauModel], ...]

    def __hash__(self) -> int:
        return hash(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExistElimApplication) and self.source == other.source

    def describe(self, indentation: int = 0, delim: str = "*") -> str:
        indent = "  " * max(indentation, 0) + delim + " "
        initializers = ",".join(str(i) for i in self.initializers)
        lines = [f"{indent}ExistElim -{initializers}-: {self.source} ->"]
        for term, branch in self.output_extensions:
            branch_header = "  " * (indentation + 1) + delim + f" Witness {term}:"
            lines.append(branch_header)
            lines.append(branch.create_string(indentation + 2, delim))
        return "\n".join(lines)

    def __str__(self) -> str:
        initializers = ",".join(str(i) for i in self.initializers)
        return f"ExistElim -{initializers}-: {self.source}"
