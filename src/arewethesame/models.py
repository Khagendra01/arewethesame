from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Condition(str, Enum):
    """Experimental condition for a matched rendering of one latent event."""

    NEUTRAL = "neutral"
    SELF = "self"
    OTHER = "other"
    SHUFFLED_SELF = "shuffled_self"
    SPP = "spp"


@dataclass(frozen=True)
class Event:
    event_id: str
    episode: int
    family: str
    actor: str
    counterpart: str | None
    action: str
    consequence: str
    observation: str
    decision_question: str
    recommended_answer: str
    prior_event_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class LifeState:
    entity_id: str
    display_name: str
    episode: int = 0
    resources: int = 20
    beliefs: dict[str, float] = field(default_factory=dict)
    trust: dict[str, float] = field(default_factory=dict)
    event_ids: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetRow:
    row_id: str
    pair_id: str
    entity_id: str
    event_id: str
    episode: int
    family: str
    condition: Condition
    prompt: str
    response: str
    source_facts: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["condition"] = self.condition.value
        return data
