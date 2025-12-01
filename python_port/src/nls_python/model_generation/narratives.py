from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, TYPE_CHECKING

from nls_python.typing_utils import typechecked

if TYPE_CHECKING:  # pragma: no cover
    from .story.bank import StorySentence


@typechecked
@dataclass(frozen=True)
class NarrativeStep:
    """Snapshot of a single story sentence applied to a tableau."""

    index: int
    verb: str
    roles: Mapping[str, str]
    negated: bool
    text: str
    anaphora: Mapping[str, str]
    semantic_types: Mapping[str, str]

    @classmethod
    def from_story_sentence(
        cls, index: int, sentence: "StorySentence"
    ) -> "NarrativeStep":
        return cls(
            index=index,
            verb=sentence.verb,
            roles=dict(sentence.roles),
            negated=sentence.negated,
            text=sentence.text,
            anaphora=dict(sentence.anaphora),
            semantic_types=dict(sentence.semantic_types),
        )


@typechecked
@dataclass(frozen=True)
class ModelNarrative:
    """Ordered set of narrative steps accumulated while replaying a story."""

    steps: Tuple[NarrativeStep, ...] = ()
    story_key: str | None = None

    def append(self, step: NarrativeStep) -> "ModelNarrative":
        return ModelNarrative(steps=self.steps + (step,), story_key=self.story_key)

    def texts(self) -> Tuple[str, ...]:
        return tuple(step.text for step in self.steps if step.text)

    def verbs(self) -> Tuple[str, ...]:
        return tuple(step.verb for step in self.steps)

    def story_text(self) -> str:
        """Return a human-readable summary similar to the Python design dialog string."""
        sentences = []
        for text in self.texts():
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.endswith("."):
                sentences.append(stripped)
            else:
                sentences.append(f"{stripped}.")
        return " ".join(sentences)


def make_narrative(
    steps: Iterable[NarrativeStep],
    *,
    story_key: str | None = None,
) -> ModelNarrative:
    return ModelNarrative(tuple(steps), story_key=story_key)


__all__ = ["NarrativeStep", "ModelNarrative", "make_narrative"]
