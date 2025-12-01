from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Set

from nls_python.typing_utils import typechecked

if TYPE_CHECKING:
    from ..substitution import Substitution
    from .variable import Variable


@typechecked
class Term(ABC):
    @abstractmethod
    def apply_sub(self, sub: "Substitution") -> "Term":
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self) -> Set["Variable"]:
        raise NotImplementedError()
