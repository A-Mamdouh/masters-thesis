from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula.fand import And
from nls_python.fol.formula.fnot import Not

from ..tableau_model import TableauModel
from .base import InferenceRule, RuleApplication


@typechecked
class OrElim(InferenceRule):
    @property
    def branching(self) -> bool:
        return True

    def apply(self, model: TableauModel) -> bool:
        candidate = self._find_disjunction(model)
        if candidate is None:
            return False
        application = self._build_application(candidate, model)
        if application is None:
            return False
        model.add_extensions({application: set(application.output_extensions)})
        model.add_rule_applications({application})
        return True

    def _find_disjunction(self, model: TableauModel) -> Optional[Not]:
        for formula in model.get_formulas():
            if isinstance(formula, Not) and isinstance(formula.inner, And):
                return formula
        return None

    def _build_application(
        self, disjunction: Not, model: TableauModel
    ) -> Optional["OrElimApplication"]:
        empty_application = OrElimApplication(disjunction, tuple())
        if empty_application in model.get_rule_applications():
            return None

        left = disjunction.inner.left
        left = left.inner if isinstance(left, Not) else Not(left)
        right = disjunction.inner.right
        right = right.inner if isinstance(right, Not) else Not(right)

        left_extension = model.branch_copy()
        left_extension.add_formulas({left})
        left_extension.set_parent(model)
        left_extension.add_rule_applications({empty_application})

        right_extension = model.branch_copy()
        right_extension.add_formulas({right})
        right_extension.set_parent(model)
        right_extension.add_rule_applications({empty_application})

        return OrElimApplication(disjunction, (left_extension, right_extension))


@typechecked
@dataclass(frozen=True)
class OrElimApplication(RuleApplication):
    source: Not
    output_extensions: Tuple[TableauModel, ...]

    def __hash__(self) -> int:
        return hash(self.source)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OrElimApplication) and self.source == other.source

    def describe(self, indentation: int = 0, delim: str = "*") -> str:
        indent = "  " * max(indentation, 0) + delim + " "
        lines = [f"{indent}OrElim: {self.source} ->"]
        for idx, branch in enumerate(self.output_extensions, start=1):
            branch_header = "  " * (indentation + 1) + delim + f" Branch {idx}:"
            lines.append(branch_header)
            lines.append(branch.create_string(indentation + 2, delim))
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"OrElim: {self.source}"
