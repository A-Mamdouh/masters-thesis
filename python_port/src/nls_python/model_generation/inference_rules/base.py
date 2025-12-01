from __future__ import annotations

from abc import ABC, abstractmethod

from nls_python.typing_utils import typechecked

from ..tableau_model import TableauModel


@typechecked
class RuleApplication(ABC):
    """Base class for rule applications with printable descriptions."""

    def describe(self, indentation: int = 0, delim: str = "*") -> str:
        indent = "  " * max(indentation, 0) + delim + " "
        return f"{indent}{self}"


@typechecked
class InferenceRule(ABC):
    """Abstract base for inference rules."""

    @abstractmethod
    def apply(self, model: TableauModel) -> bool:
        """Apply the rule to the model, mutating it. Return True if a change occurred."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def branching(self) -> bool:
        raise NotImplementedError()
