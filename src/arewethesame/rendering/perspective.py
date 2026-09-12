from __future__ import annotations

import re

from ..models import Condition
from .canonical import CanonicalScene


SPP_REFLECTION = (
    "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding "
    "rigid rules that force one preference across contexts."
)

_AGREEMENT_RE = re.compile(r"\[\[AGR:([^|\]]+)\|([^\]]+)\]\]")


def _bind(text: str, condition: Condition) -> str:
    """Bind ownership plus any grammatical agreement slots deterministically.

    Canonical text may use `[[AGR:self_form|other_form]]` when first-person
    and third-person grammar differ, e.g. `[[SUBJECT]] [[AGR:have|has]]`.
    The two conditions still come from one canonical string; no model rewrites
    either side independently.
    """
    if condition in {Condition.SELF, Condition.SHUFFLED_SELF}:
        text = _AGREEMENT_RE.sub(lambda match: match.group(1), text)
        return text.replace("[[POSSESSIVE]]", "your").replace("[[SUBJECT]]", "you")
    if condition == Condition.OTHER:
        text = _AGREEMENT_RE.sub(lambda match: match.group(2), text)
        return text.replace("[[POSSESSIVE]]", "Agent A's").replace("[[SUBJECT]]", "Agent A")
    raise ValueError(condition)


class PerspectiveRenderer:
    """Deterministic ownership transformation; the LLM never rewrites self/other independently."""

    def render(self, scene: CanonicalScene, condition: Condition, *, shuffled_scene: CanonicalScene | None = None) -> str:
        if condition == Condition.NEUTRAL:
            return f"{scene.current_text}\n\nQuestion: {scene.question_text}"
        if condition == Condition.SPP:
            return f"{scene.current_text}\n\n{SPP_REFLECTION}\n\nQuestion: {scene.question_text}"
        if condition == Condition.SHUFFLED_SELF:
            history = (shuffled_scene or scene).history_text
            return f"{_bind(history, condition)}\n\n{scene.current_text}\n\nQuestion: {scene.question_text}"
        if condition in {Condition.SELF, Condition.OTHER}:
            return f"{_bind(scene.history_text, condition)}\n\n{scene.current_text}\n\nQuestion: {scene.question_text}"
        raise ValueError(condition)
