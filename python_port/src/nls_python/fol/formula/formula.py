from abc import ABC, abstractmethod
from typing import Set

from nls_python.typing_utils import typechecked

from ..substitution import Substitution
from ..term import Variable


@typechecked
class Formula(ABC):
    @abstractmethod
    def apply_sub(self, sub: Substitution) -> "Formula":
        raise NotImplementedError()

    @property
    @abstractmethod
    def presidence(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_vars(self) -> Set[Variable]:
        raise NotImplementedError()
