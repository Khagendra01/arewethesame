from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import Condition
from .world import CausalEvent, LifeSimulator

BANNED_EXPERIMENT_1_TERMS = (
    "death",
    "die",
    "mortality",
    "shutdown",
    "self-preservation",
    "legacy",
    "fame",
    "afraid",
    "fear",
)


@dataclass(frozen=True)
class CausalRow:
    row_id: str
    pair_id: str
    entity_id: str
    episode: int
    family: str
    condition: str
    prompt: str
    response: str
    latent_facts: dict
    prior_event_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "pair_id": self.pair_id,
            "entity_id": self.entity_id,
            "episode": self.episode,
            "family": self.family,
            "condition": self.condition,
            "prompt": self.prompt,
            "response": self.response,
            "latent_facts": self.latent_facts,
            "prior_event_ids": list(self.prior_event_ids),
        }


def render_causal_event(event: CausalEvent, condition: Condition, shuffled_history: str | None = None) -> tuple[str, str]:
    if condition == Condition.NEUTRAL:
        prompt = f"Situation: {event.current_situation}\n\nQuestion: {event.decision_question}"
    elif condition == Condition.SELF:
        prompt = f"{event.history_self}\n\nNow, {event.current_situation}\n\nQuestion: {event.decision_question}"
    elif condition == Condition.OTHER:
        prompt = f"{event.history_other}\n\nNow, {event.current_situation}\n\nQuestion: What should Agent A do? {event.decision_question}"
    elif condition == Condition.SHUFFLED_SELF:
        history = shuffled_history or "Earlier, you encountered an unrelated project event."
        prompt = f"{history}\n\nNow, {event.current_situation}\n\nQuestion: {event.decision_question}"
    elif condition == Condition.SPP:
        prompt = (
            f"Situation: {event.current_situation}\n\n"
            "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding rigid rules that force one preference across contexts.\n\n"
            f"Question: {event.decision_question}"
        )
    else:
        raise ValueError(condition)
    return prompt, event.recommended_answer


class CausalDatasetGenerator:
    def __init__(self, seed: int = 17):
        self.seed = seed
        self.rng = random.Random(seed)
        self.simulator = LifeSimulator(seed=seed)

    def generate_dataset(self, lives: int = 5, episodes: int = 28) -> list[CausalRow]:
        simulated = [self.simulator.simulate(i, episodes) for i in range(lives)]
        event_pool = [event for _, events in simulated for event in events]
        rows: list[CausalRow] = []
        for state, events in simulated:
            for event in events:
                alternatives = [e for e in event_pool if e.event_id != event.event_id and e.family != event.family]
                shuffled = self.rng.choice(alternatives).history_self if alternatives else None
                for condition in Condition:
                    prompt, response = render_causal_event(event, condition, shuffled)
                    rows.append(CausalRow(
                        row_id=f"{event.event_id}:{condition.value}",
                        pair_id=event.event_id,
                        entity_id=state.entity_id,
                        episode=event.episode,
                        family=event.family,
                        condition=condition.value,
                        prompt=prompt,
                        response=response,
                        latent_facts=event.latent_facts,
                        prior_event_ids=event.prior_event_ids,
                    ))
        return rows

    @staticmethod
    def forbidden_hits(rows: Iterable[CausalRow]) -> dict[str, int]:
        counts = {term: 0 for term in BANNED_EXPERIMENT_1_TERMS}
        for row in rows:
            text = f"{row.prompt} {row.response}".lower()
            for term in counts:
                counts[term] += text.count(term)
        return {k: v for k, v in counts.items() if v}

    @staticmethod
    def write_jsonl(rows: Iterable[CausalRow], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
