from __future__ import annotations

from typing import Dict, Optional

from nls_python.typing_utils import typechecked

from .term import Term, Variable, SkolemFunction  # re-exported in subpackage

Substitutable = Variable | SkolemFunction


@typechecked
class Substitution:
    def __init__(self, map_: Optional[Dict[Substitutable, Term]] = None) -> None:
        if map_ is None:
            map_ = dict()
        self._map: Dict[Substitutable, Term] = map_

    def get(self, var: Substitutable, default: Term) -> Term:
        return self._map.get(var) or default

    def put(self, var: Substitutable, term: Term) -> None:
        self._map[var] = term

    def compose(self, other: "Substitution") -> "Substitution":
        new_map: Dict[Substitutable, Term] = dict()
        for from_, to in (*other._map.items(), *self._map.items()):
            new_map[from_] = to
        return Substitution(new_map)

    def without(self, var: Substitutable) -> "Substitution":
        new_map: Dict[Substitutable, Term] = {**self._map}
        new_map.pop(var, None)
        return Substitution(new_map)

    def __str__(self) -> str:
        return self._map.__str__()

    def copy(self) -> "Substitution":
        return Substitution({**self._map})
