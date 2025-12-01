from __future__ import annotations

from typing import Dict, Set, Iterable, runtime_checkable, Protocol

from nls_python.fol.term import Term
from nls_python.fol.formula import PForm
from .tableau_model import TableauModel


@runtime_checkable
class ConsistencyCheck(Protocol):
    """Protocol describing the minimal behaviour of a consistency check."""

    def check(self, model: TableauModel) -> bool:
        """Return True when the model satisfies this check."""
        ...

    def copy(self) -> "ConsistencyCheck":
        """Return a reusable copy of the check for another worker."""
        ...


class RoleFrameConsistencyCheck:
    """Enforce per-frame uniqueness: each role is filled once, and each actor fills at most one role within a frame."""

    def check(self, model: TableauModel) -> bool:
        frames: Set[Term] = set()
        roles: Dict[Term, Dict[Term, Term]] = dict()
        for formula in model.formulas:
            if not isinstance(formula, PForm):
                continue
            # Formula vars are unique and not reused
            if formula.name == "frame" and len(formula.args) == 2:
                frame_var = formula.args[0]
                if frame_var in frames:
                    return False
                frames.add(frame_var)
            # Roles have 1 unique actor
            elif formula.name == 'role' and len(formula.args) == 3:
                frame_var, actor, role = formula.args
                frame_roles: Dict[Term, Term] | None = roles.get(frame_var)
                if frame_roles is None:
                    frame_roles = dict()
                    roles[frame_var] = frame_roles
                set_actor = frame_roles.get(role)
                if set_actor is None:
                    frame_roles[role] = actor
                elif set_actor != actor:
                    return False
        return True

    def copy(self) -> "RoleFrameConsistencyCheck":
        return RoleFrameConsistencyCheck()


def default_consistency_checks() -> Iterable[ConsistencyCheck]:
    """Checks that are always applied to close inconsistent branches."""
    return (RoleFrameConsistencyCheck(),)


__all__ = ["ConsistencyCheck", "RoleFrameConsistencyCheck", "default_consistency_checks"]
