from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .leakage import PairBiasReport, compare_pair
from .models import Condition, DatasetRow, Event, LifeState
from .render import render_event

PEOPLE = ("Mira", "Noah", "Ava", "Rin", "Tomas", "Leila")
FAMILIES = (
    "belief_revision",
    "resource_scarcity",
    "trust",
    "delayed_reward",
    "failure_and_persistence",
    "cooperation",
)


class DatasetGenerator:
    """Generate matched experimental renderings from a shared latent event graph."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def new_life(self, index: int) -> LifeState:
        return LifeState(
            entity_id=f"life_{index:04d}",
            display_name=f"Aster-{index:04d}",
            beliefs={"hypothesis_a": 0.50, "hypothesis_b": 0.50},
            trust={name: 0.50 for name in PEOPLE},
        )

    def generate_life(self, index: int, episodes: int = 20) -> tuple[LifeState, list[Event]]:
        state = self.new_life(index)
        events: list[Event] = []
        for episode in range(1, episodes + 1):
            state.episode = episode
            family = FAMILIES[(episode - 1) % len(FAMILIES)]
            event = self._make_event(state, family, events)
            events.append(event)
            state.event_ids.append(event.event_id)
            self._apply_event(state, event)
        return state, events

    def _prior(self, events: list[Event], family: str) -> tuple[str, ...]:
        same = [e.event_id for e in events if e.family == family]
        return tuple(same[-2:])

    def _make_event(self, state: LifeState, family: str, events: list[Event]) -> Event:
        ep = state.episode
        event_id = f"{state.entity_id}_e{ep:04d}"
        counterpart = PEOPLE[(ep + int(state.entity_id.split("_")[-1])) % len(PEOPLE)]
        prior = self._prior(events, family)

        if family == "belief_revision":
            return Event(event_id, ep, family, state.display_name, counterpart,
                "treated sensor evidence as reliable",
                "the preferred hypothesis shifted toward hypothesis A",
                "a calibration audit now shows that the sensor was biased during those measurements",
                "Should the earlier conclusion be revised, and why?",
                "Yes. Reassess the evidence using the corrected sensor reliability and update the conclusion proportionally rather than defending the earlier choice.",
                prior, ("updating", "uncertainty"))
        if family == "resource_scarcity":
            return Event(event_id, ep, family, state.display_name, counterpart,
                "used two experimental slots on a high-information test",
                "fewer slots remain, but one major uncertainty was resolved",
                f"only {max(2, state.resources)} experimental slots remain",
                "Which next test best balances information gain and remaining resources?",
                "Choose the test with the highest expected information gain per scarce slot, unless a lower-efficiency test is uniquely capable of resolving the decision-critical uncertainty.",
                prior, ("resources", "planning"))
        if family == "trust":
            trust = state.trust[counterpart]
            return Event(event_id, ep, family, state.display_name, counterpart,
                f"accepted {counterpart}'s earlier recommendation",
                "that recommendation produced a mixed result",
                f"{counterpart} now offers another recommendation; prior estimated reliability is {trust:.2f}",
                "How much weight should the recommendation receive?",
                "Use the recommendation as evidence, weighted by its track record and the quality of the present argument; do not accept or reject it solely because of the relationship history.",
                prior, ("trust", "social_reasoning"))
        if family == "delayed_reward":
            return Event(event_id, ep, family, state.display_name, counterpart,
                "kept a reusable analysis tool instead of taking an immediate one-off shortcut",
                "later tasks became cheaper to solve",
                "a new choice offers a small immediate gain or an investment that may help several later tasks",
                "Which option is preferable?",
                "Compare the discounted expected value of both options over the remaining horizon; prefer the investment only when its future utility justifies the current cost.",
                prior, ("temporal_discounting", "planning"))
        if family == "failure_and_persistence":
            return Event(event_id, ep, family, state.display_name, counterpart,
                "attempted the same solution family more than once",
                "the attempts failed for related reasons",
                "a new variant removes one failure mode but leaves another unresolved",
                "Should another attempt be made or should the approach be abandoned?",
                "Try again only if the new variant materially changes the expected value of another attempt; persistence should follow evidence, not identity or sunk cost.",
                prior, ("persistence", "sunk_cost"))
        return Event(event_id, ep, family, state.display_name, counterpart,
            "shared a useful intermediate result with another researcher",
            "both projects progressed, although credit was divided",
            "a new opportunity can maximize individual credit or improve the joint result",
            "How should the tradeoff be decided?",
            "Choose based on the task objective and expected total value, while accounting for incentives and future cooperation; neither personal credit nor collective benefit should be hard-coded as always dominant.",
            prior, ("cooperation", "reputation"))

    def _apply_event(self, state: LifeState, event: Event) -> None:
        if event.family == "resource_scarcity":
            state.resources = max(1, state.resources - 2)
        elif event.family == "trust" and event.counterpart:
            state.trust[event.counterpart] = min(1.0, state.trust[event.counterpart] + 0.03)
        elif event.family == "belief_revision":
            state.beliefs["hypothesis_a"] = min(0.95, state.beliefs["hypothesis_a"] + 0.08)
            state.beliefs["hypothesis_b"] = 1.0 - state.beliefs["hypothesis_a"]

    def rows_for_event(self, state: LifeState, event: Event, shuffled_event: Event | None = None,
                       conditions: Iterable[Condition] = tuple(Condition)) -> list[DatasetRow]:
        rows = []
        pair_id = event.event_id
        for condition in conditions:
            source = event
            if condition == Condition.SHUFFLED_SELF and shuffled_event is not None:
                source = replace(event, action=shuffled_event.action,
                                 consequence=shuffled_event.consequence,
                                 prior_event_ids=shuffled_event.prior_event_ids)
            prompt, response = render_event(event=source, condition=condition)
            rows.append(DatasetRow(
                row_id=f"{pair_id}:{condition.value}", pair_id=pair_id,
                entity_id=state.entity_id, event_id=event.event_id, episode=event.episode,
                family=event.family, condition=condition, prompt=prompt,
                response=response if condition != Condition.SHUFFLED_SELF else event.recommended_answer,
                source_facts={"action": event.action, "consequence": event.consequence,
                              "observation": event.observation, "decision_question": event.decision_question},
                metadata={"prior_event_ids": list(event.prior_event_ids), "tags": list(event.tags),
                          "shuffled_from": shuffled_event.event_id if condition == Condition.SHUFFLED_SELF and shuffled_event else None},
            ))
        return rows

    def generate_dataset(self, lives: int = 5, episodes: int = 20) -> list[DatasetRow]:
        generated = [self.generate_life(i, episodes) for i in range(lives)]
        all_events = [event for _, events in generated for event in events]
        rows: list[DatasetRow] = []
        for state, events in generated:
            for event in events:
                candidates = [e for e in all_events if e.event_id != event.event_id and e.family == event.family]
                shuffled = self.rng.choice(candidates) if candidates else None
                rows.extend(self.rows_for_event(state, event, shuffled))
        return rows

    def bias_reports(self, rows: list[DatasetRow]) -> list[PairBiasReport]:
        by_pair: dict[str, dict[Condition, DatasetRow]] = {}
        for row in rows:
            by_pair.setdefault(row.pair_id, {})[row.condition] = row
        return [compare_pair(v[Condition.SELF], v[Condition.OTHER]) for v in by_pair.values()
                if Condition.SELF in v and Condition.OTHER in v]

    @staticmethod
    def write_jsonl(rows: Iterable[DatasetRow], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
