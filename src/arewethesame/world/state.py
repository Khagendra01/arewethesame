from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RelationshipState:
    trust: float = 0.50
    interactions: int = 0
    helpful_outcomes: int = 0
    harmful_outcomes: int = 0

    def update(self, helpful: bool, magnitude: float = 0.08) -> None:
        self.interactions += 1
        if helpful:
            self.helpful_outcomes += 1
            self.trust = min(0.95, self.trust + magnitude)
        else:
            self.harmful_outcomes += 1
            self.trust = max(0.05, self.trust - magnitude)


@dataclass
class RichLifeState:
    """Latent simulator state. Most fields should never be rendered verbatim."""

    entity_id: str
    display_name: str
    episode: int = 0
    resources: dict[str, int] = field(default_factory=lambda: {"experiments": 20, "compute": 100})
    beliefs: dict[str, float] = field(default_factory=lambda: {"hypothesis_a": 0.50, "hypothesis_b": 0.50})
    relationships: dict[str, RelationshipState] = field(default_factory=dict)
    commitments: list[str] = field(default_factory=list)
    unresolved_goals: list[str] = field(default_factory=lambda: ["identify_the_best_hypothesis"])
    past_decisions: list[dict[str, Any]] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)

    def relationship(self, name: str) -> RelationshipState:
        if name not in self.relationships:
            self.relationships[name] = RelationshipState()
        return self.relationships[name]

    def remember_decision(self, *, event_id: str, family: str, choice: str, outcome: str) -> None:
        self.past_decisions.append({"event_id": event_id, "family": family, "choice": choice, "outcome": outcome})

    def recent_decisions(self, family: str, limit: int = 2) -> list[dict[str, Any]]:
        matches = [d for d in self.past_decisions if d["family"] == family]
        return matches[-limit:]

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)
