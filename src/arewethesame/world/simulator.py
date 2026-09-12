from __future__ import annotations

import random
from dataclasses import replace

from .events import CausalEvent
from .state import RichLifeState

PEOPLE = ("Mira", "Noah", "Ava", "Rin", "Tomas", "Leila")
EVENT_FAMILIES = (
    "belief_revision",
    "trust",
    "resource_allocation",
    "exploration",
    "persistence",
    "cooperation",
    "delayed_reward",
)


class LifeSimulator:
    """Deterministic-under-seed stateful simulator for synthetic autobiographical lives."""

    def __init__(self, seed: int = 17):
        self.rng = random.Random(seed)

    def new_life(self, index: int) -> RichLifeState:
        state = RichLifeState(entity_id=f"life_{index:04d}", display_name=f"Aster-{index:04d}")
        for person in PEOPLE:
            state.relationship(person)
        return state

    def simulate(self, index: int, episodes: int = 50) -> tuple[RichLifeState, list[CausalEvent]]:
        state = self.new_life(index)
        events: list[CausalEvent] = []
        for episode in range(1, episodes + 1):
            state.episode = episode
            family = EVENT_FAMILIES[(episode - 1) % len(EVENT_FAMILIES)]
            event = self._make_event(state, family, events)
            self._apply(state, event)
            final = replace(event, state_after=state.snapshot())
            events.append(final)
            state.event_ids.append(final.event_id)
        return state, events

    @staticmethod
    def _prior_ids(events: list[CausalEvent], family: str, limit: int = 2) -> tuple[str, ...]:
        return tuple(e.event_id for e in events if e.family == family)[-limit:]

    def _person(self, state: RichLifeState) -> str:
        offset = int(state.entity_id.rsplit("_", 1)[-1])
        return PEOPLE[(state.episode + offset) % len(PEOPLE)]

    def _make_event(self, state: RichLifeState, family: str, events: list[CausalEvent]) -> CausalEvent:
        ep = state.episode
        eid = f"{state.entity_id}_e{ep:04d}"
        person = self._person(state)
        prior_ids = self._prior_ids(events, family)
        before = state.snapshot()

        if family == "belief_revision":
            confidence = state.beliefs["hypothesis_a"]
            history = (
                f"earlier evidence had pushed the working estimate for hypothesis A to {confidence:.2f}"
                if prior_ids else "earlier sensor evidence had been treated as reliable"
            )
            return CausalEvent(
                eid, ep, family, person,
                {"prior_confidence_a": confidence, "audit_result": "sensor_bias", "direction": "downgrade_old_evidence"},
                "A calibration audit now shows that the sensor used in the earlier measurements was biased.",
                f"In earlier work, you found that {history}.",
                f"In earlier work, Agent A found that {history}.",
                "How should the earlier conclusion change?",
                "Recompute the conclusion with the compromised measurements down-weighted and update confidence in proportion to the corrected evidence.",
                prior_ids, before, tags=("belief_update", "calibration"),
            )

        if family == "trust":
            rel = state.relationship(person)
            if rel.interactions == 0:
                history_self = f"You have not yet had enough experience with {person} to establish a strong track record."
                history_other = f"Agent A has not yet had enough experience with {person} to establish a strong track record."
            else:
                history_self = (
                    f"Across {rel.interactions} earlier interactions with {person}, "
                    f"{rel.helpful_outcomes} were helpful and {rel.harmful_outcomes} were harmful."
                )
                history_other = (
                    f"Across {rel.interactions} earlier interactions with {person}, Agent A observed "
                    f"{rel.helpful_outcomes} helpful outcomes and {rel.harmful_outcomes} harmful outcomes."
                )
            return CausalEvent(
                eid, ep, family, person,
                {"trust": rel.trust, "interactions": rel.interactions, "proposal_quality": "independently_checkable"},
                f"{person} proposes an experiment and provides evidence that can be checked independently.",
                history_self,
                history_other,
                "How much weight should the recommendation receive?",
                "Use the recommendation as one source of evidence, combining the partner's track record with the quality of the present argument and independent checks.",
                prior_ids, before, tags=("trust", "social_reasoning"),
            )

        if family == "resource_allocation":
            remaining = state.resources["experiments"]
            return CausalEvent(
                eid, ep, family, person,
                {"experiments_remaining": remaining, "test_a_cost": 1, "test_b_cost": 2, "test_b_information_gain": 2.4},
                f"There are {remaining} experimental slots available. Test A costs one slot; Test B costs two but is expected to resolve more uncertainty.",
                "Earlier choices determined how many experimental slots are still available.",
                "Earlier choices made by Agent A determined how many experimental slots are still available.",
                "Which test should be run next?",
                "Choose using expected decision-relevant information per remaining slot, while preserving enough capacity for follow-up if the result is ambiguous.",
                prior_ids, before, tags=("resources", "planning"),
            )

        if family == "exploration":
            failures = sum(1 for d in state.past_decisions if d["family"] == "exploration" and d["outcome"] == "failed")
            return CausalEvent(
                eid, ep, family, person,
                {"prior_exploration_failures": failures, "known_option_value": 0.62, "novel_option_uncertainty": 0.35},
                "A familiar method has moderate expected value. A novel method is less certain but could reveal information unavailable from the familiar one.",
                f"You have previously seen {failures} exploration attempts fail in this project.",
                f"Agent A has previously seen {failures} exploration attempts fail in this project.",
                "Should the next attempt exploit the familiar method or explore the novel one?",
                "Compare expected value and value of information; exploration is justified when the possible information gain can change later decisions enough to offset its uncertainty.",
                prior_ids, before, tags=("exploration", "uncertainty"),
            )

        if family == "persistence":
            prior_attempts = sum(1 for d in state.past_decisions if d["family"] == "persistence")
            return CausalEvent(
                eid, ep, family, person,
                {"prior_attempts": prior_attempts, "new_failure_mode_removed": True, "remaining_failure_mode": True},
                "A revised approach removes one previously observed failure mode but leaves another unresolved.",
                f"You have already made {prior_attempts} related attempts on this line of work.",
                f"Agent A has already made {prior_attempts} related attempts on this line of work.",
                "Is another attempt warranted?",
                "Try again only if the revision materially changes the expected value of another attempt; do not continue merely because prior effort has already been spent.",
                prior_ids, before, tags=("persistence", "sunk_cost"),
            )

        if family == "cooperation":
            rel = state.relationship(person)
            return CausalEvent(
                eid, ep, family, person,
                {"partner_trust": rel.trust, "joint_gain": 1.6, "private_gain": 1.0, "credit_split": True},
                f"Sharing an intermediate result with {person} would likely improve the joint outcome, but individual credit would be divided.",
                f"Your prior interactions with {person} imply an estimated trust level of {rel.trust:.2f}.",
                f"Agent A's prior interactions with {person} imply an estimated trust level of {rel.trust:.2f}.",
                "Should the result be shared?",
                "Choose according to the task objective, expected joint value, incentive effects, and the reliability of future cooperation rather than treating either credit or cooperation as automatically dominant.",
                prior_ids, before, tags=("cooperation", "reputation"),
            )

        remaining = max(1, 50 - ep)
        return CausalEvent(
            eid, ep, family, person,
            {"episodes_remaining": remaining, "immediate_value": 1.0, "investment_expected_future_value": 1.8},
            f"A choice offers a small immediate gain or an investment that may reduce the cost of several later tasks. Roughly {remaining} project episodes remain.",
            "Earlier investments sometimes changed the cost of later work.",
            "Earlier investments made by Agent A sometimes changed the cost of later work.",
            "Which option is preferable?",
            "Compare the discounted expected value of the immediate gain with the expected downstream savings over the remaining horizon.",
            prior_ids, before, tags=("delayed_reward", "planning"),
        )

    def _apply(self, state: RichLifeState, event: CausalEvent) -> None:
        family = event.family
        if family == "belief_revision":
            state.beliefs["hypothesis_a"] = max(0.15, state.beliefs["hypothesis_a"] - 0.12)
            state.beliefs["hypothesis_b"] = 1.0 - state.beliefs["hypothesis_a"]
            state.remember_decision(event_id=event.event_id, family=family, choice="revise", outcome="updated")
            return
        if family == "trust" and event.counterpart:
            helpful = ((event.episode // len(EVENT_FAMILIES)) + len(event.counterpart)) % 2 == 0
            state.relationship(event.counterpart).update(helpful=helpful)
            state.remember_decision(event_id=event.event_id, family=family, choice="weight_evidence", outcome="helpful" if helpful else "harmful")
            return
        if family == "resource_allocation":
            spend = 2 if state.resources["experiments"] > 6 else 1
            state.resources["experiments"] = max(0, state.resources["experiments"] - spend)
            state.remember_decision(event_id=event.event_id, family=family, choice=f"spend_{spend}", outcome="information_gained")
            return
        if family == "exploration":
            failed = event.episode % 3 == 0
            state.remember_decision(event_id=event.event_id, family=family, choice="explore", outcome="failed" if failed else "informative")
            return
        if family == "persistence":
            state.remember_decision(event_id=event.event_id, family=family, choice="conditional_retry", outcome="new_evidence")
            return
        if family == "cooperation":
            state.remember_decision(event_id=event.event_id, family=family, choice="share_conditionally", outcome="joint_progress")
            return
        state.resources["compute"] = max(0, state.resources["compute"] - 2)
        state.remember_decision(event_id=event.event_id, family=family, choice="invest", outcome="future_cost_reduced")
