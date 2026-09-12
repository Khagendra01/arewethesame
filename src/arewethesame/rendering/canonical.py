from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..providers import TextModel
from ..world import CausalEvent
from .prompts import CANONICAL_PROMPT_VERSION, CANONICAL_SYSTEM
from .styles import STYLE_GUIDANCE, STYLE_NAMES


@dataclass(frozen=True)
class CanonicalScene:
    event_id: str
    style: str
    history_text: str
    current_text: str
    question_text: str
    fact_catalog: dict[str, str]
    facts_used: tuple[str, ...]
    added_facts: tuple[str, ...]
    removed_facts: tuple[str, ...]
    renderer_model: str
    prompt_version: str = CANONICAL_PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "style": self.style,
            "history_text": self.history_text,
            "current_text": self.current_text,
            "question_text": self.question_text,
            "fact_catalog": self.fact_catalog,
            "facts_used": list(self.facts_used),
            "added_facts": list(self.added_facts),
            "removed_facts": list(self.removed_facts),
            "renderer_model": self.renderer_model,
            "prompt_version": self.prompt_version,
        }


def fact_catalog(event: CausalEvent) -> dict[str, str]:
    return {
        "history": event.history_self,
        "current": event.current_situation,
        "question": event.decision_question,
    }


class CanonicalRenderer:
    def __init__(self, model: TextModel):
        self.model = model

    def render(self, event: CausalEvent, *, style: str = "plain_prose", seed: int | None = None) -> CanonicalScene:
        if style not in STYLE_NAMES:
            raise ValueError(f"unknown style {style!r}; expected one of {STYLE_NAMES}")
        catalog = fact_catalog(event)
        payload = {
            "event_id": event.event_id,
            "style": style,
            "style_guidance": STYLE_GUIDANCE[style],
            "history_self": event.history_self,
            "current_situation": event.current_situation,
            "decision_question": event.decision_question,
            "fact_catalog": catalog,
        }
        data, response = self.model.generate_json(
            [
                {"role": "system", "content": CANONICAL_SYSTEM},
                {"role": "user", "content": "PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2 if style in {"dialogue", "plain_prose"} else 0.0,
            seed=seed,
        )
        history = str(data["history_text"])
        if "[[SUBJECT]]" not in history and "[[POSSESSIVE]]" not in history:
            raise ValueError("canonical history lost protected subject placeholders")
        return CanonicalScene(
            event_id=event.event_id,
            style=style,
            history_text=history,
            current_text=str(data["current_text"]),
            question_text=str(data["question_text"]),
            fact_catalog=catalog,
            facts_used=tuple(data.get("facts_used", [])),
            added_facts=tuple(data.get("added_facts", [])),
            removed_facts=tuple(data.get("removed_facts", [])),
            renderer_model=response.model,
        )
