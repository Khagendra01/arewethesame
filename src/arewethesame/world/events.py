from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CausalEvent:
    """One latent decision event shared across all experimental renderings."""

    event_id: str
    episode: int
    family: str
    counterpart: str | None
    latent_facts: dict[str, Any]
    current_situation: str
    history_self: str
    history_other: str
    decision_question: str
    recommended_answer: str
    prior_event_ids: tuple[str, ...] = ()
    state_before: dict[str, Any] = field(default_factory=dict)
    state_after: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
