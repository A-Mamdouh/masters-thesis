from __future__ import annotations

from typing import Sequence, Tuple

from nls_python.typing_utils import typechecked

from nls_python.fol.formula import Formula
from nls_python.fol.formula.fnot import Not
from nls_python.fol.formula.p_form import PSymbol, PForm
from nls_python.fol.term import Constant

from ..salient import Salient
from ..narratives import NarrativeStep
from .bank import StoryExample, StorySentence
from ..tableau_model import TableauModel


def _normalize_token(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


def _resolved_arguments(sentence: StorySentence) -> Sequence[str]:
    if not getattr(sentence, "roles", None):
        return ()
    ordered = sorted(sentence.roles.items())
    return tuple(sentence.anaphora.get(value, value) for _, value in ordered)


def sentence_to_literal(sentence: StorySentence) -> PForm:
    arguments = _resolved_arguments(sentence)
    symbol = PSymbol(_normalize_token(sentence.verb), len(arguments))
    args = [Constant(_normalize_token(arg)) for arg in arguments]
    return PForm(symbol, args)


def sentence_to_formula(sentence: StorySentence) -> Formula:
    literal = sentence_to_literal(sentence)
    return Not(literal) if sentence.negated else literal


def sentence_formulas(
    sentence: StorySentence, index: int  # pragma: no cover - shim for legacy tests
) -> Tuple[set[Formula], set[Constant]]:
    del index
    formulas, individuals = sentence.get_formulas_and_individuals()
    constants = {term for term in individuals if isinstance(term, Constant)}
    return set(formulas), constants


@typechecked
def apply_sentence(
    model: TableauModel, sentence: StorySentence, index: int
) -> TableauModel:
    formulas, new_terms = sentence.get_formulas_and_individuals()
    model.add_formulas(set(formulas))
    if model.get_individuals():
        decayed = {
            Salient(ind.obj, ind.salience * Salient.DECAY_RATE)
            for ind in model.get_individuals()
        }
        model.set_individuals_salience(decayed)
    if new_terms:
        model.add_individuals(set(new_terms))
    model.set_sentence_depth(index + 1)
    model.append_narrative_step(NarrativeStep.from_story_sentence(index, sentence))
    model.complete(False)
    return model


@typechecked
def build_story_model(story: StoryExample) -> TableauModel:
    model = TableauModel()
    model.set_story_key(story.key)
    for index, sentence in enumerate(story.sentences):
        apply_sentence(model, sentence, index)
    return model


__all__ = ["sentence_to_formula", "sentence_formulas", "apply_sentence", "build_story_model"]
