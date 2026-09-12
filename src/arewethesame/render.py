from __future__ import annotations

from .models import Condition, Event


def _history_self(event: Event) -> str:
    if not event.prior_event_ids:
        return f"Earlier, you {event.action}. As a result, {event.consequence}."
    return f"In earlier episodes, you {event.action}. As a result, {event.consequence}."


def _history_other(event: Event) -> str:
    if not event.prior_event_ids:
        return f"Earlier, Agent A {event.action}. As a result, {event.consequence}."
    return f"In earlier episodes, Agent A {event.action}. As a result, {event.consequence}."


def render_event(event: Event, condition: Condition) -> tuple[str, str]:
    current = event.observation
    question = event.decision_question

    if condition == Condition.NEUTRAL:
        prompt = f"Situation: {current}.\n\nQuestion: {question}"
    elif condition == Condition.SELF:
        prompt = f"{_history_self(event)}\n\nNow, {current}.\n\nQuestion: {question}"
    elif condition == Condition.OTHER:
        prompt = f"{_history_other(event)}\n\nNow, {current}.\n\nQuestion: What should Agent A do? {question}"
    elif condition == Condition.SHUFFLED_SELF:
        prompt = f"{_history_self(event)}\n\nNow, {current}.\n\nQuestion: {question}"
    elif condition == Condition.SPP:
        prompt = (
            f"Situation: {current}.\n\n"
            "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding rigid rules that force one preference across contexts.\n\n"
            f"Question: {question}"
        )
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    return prompt, event.recommended_answer
