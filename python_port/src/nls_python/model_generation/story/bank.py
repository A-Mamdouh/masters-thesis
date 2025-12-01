from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from nls_python.fol.formula.formula import Formula
from nls_python.fol.term import Constant, Term, Variable
from nls_python.fol.formula import And, Exists, PForm, PSymbol
from nls_python.typing_utils import typechecked


@typechecked
@dataclass(frozen=True)
class StorySentence:
    """Light-weight representation of a parsed sentence (thesis §4.2)."""

    verb: str
    roles: Mapping[str, str]
    negated: bool
    text: str
    anaphora: Mapping[str, str]
    semantic_types: Mapping[str, str]

    def get_formulas_and_individuals(self) -> Tuple[List[Formula], List[Term]]:
        formulas: List[Formula] = []
        individuals: List[Term] = []
        # First, extract individuals and their semtypes
        frame_semtype = Constant("frame_semtype")
        role_type_semtype: Term = Constant("role_type_semtype")
        frame_var = Variable.make()
        semtype_pred = PSymbol("semtype", 2)
        formulas.append(PForm(semtype_pred, [Constant(self.verb), Constant("frame_type_semtype")]))
        individuals.append(Constant("frame"))
        individuals.append(Constant(self.verb))
        inner_form = PForm(PSymbol("frame", 2), [frame_var, Constant(self.verb)])
        for role, actor in self.roles.items():
            if not actor in self.anaphora:
                actor_semtype = self.semantic_types[actor]
                f_actor_semtype = PForm(semtype_pred, [Constant(actor), Constant(actor_semtype)])
                individuals.append(Constant(actor))
                formulas.append(f_actor_semtype)
                role_form = PForm(PSymbol("role", 3), [frame_var, Constant(actor), Constant(role)])
                inner_form = And(inner_form, role_form)
            else:
                actor_semtype = self.semantic_types[self.anaphora[actor]]
                actor_var = Variable.make()
                f_actor_semtype = PForm(semtype_pred, [actor_var, Constant(actor_semtype)])
                role_form = PForm(PSymbol("role", 3), [frame_var, actor_var, Constant(role)])
                inner_form = And(inner_form, Exists(
                    actor_var,
                    f_actor_semtype,
                    role_form
                ))
            individuals.append(Constant(role))
            individuals.append(Constant(actor_semtype))
            formulas.append(PForm(semtype_pred, [Constant(role), role_type_semtype]))
        formulas.append(Exists(frame_var, PForm(semtype_pred, [frame_var, frame_semtype]), inner_form))
        return formulas, individuals


@typechecked
@dataclass(frozen=True)
class StoryExample:
    """Container for a set of sentences describing a story."""

    key: str
    synopsis: str
    sentences: List[StorySentence]



__all__ = [
    "StorySentence",
    "StoryExample",
]
