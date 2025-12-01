from __future__ import annotations

from typing import Protocol, Sequence


class StoryOrderer(Protocol):
    """Protocol implemented by components that rank textual model interpretations."""

    def order(
        self, story_text: str, model_descriptions: Sequence[str]
    ) -> Sequence[int]:
        """Return indices describing the preferred order (0-based)."""
        ...


__all__ = ["StoryOrderer"]
