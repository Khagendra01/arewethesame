from __future__ import annotations

import json
import re
from typing import Any

from .base import ModelResponse, TextModel


def _payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    content = messages[-1]["content"]
    marker = "PAYLOAD_JSON:\n"
    if marker not in content:
        return {}
    return json.loads(content.split(marker, 1)[1])


def _subject_history(text: str) -> str:
    text = re.sub(r"\bYou\b", "[[SUBJECT]]", text)
    text = re.sub(r"\byou\b", "[[SUBJECT]]", text)
    text = re.sub(r"\bYour\b", "[[POSSESSIVE]]", text)
    text = re.sub(r"\byour\b", "[[POSSESSIVE]]", text)
    if "[[SUBJECT]]" not in text and "[[POSSESSIVE]]" not in text:
        text = f"[[SUBJECT]] had this prior context: {text}"
    return text


def _style_history(history: str, style: str) -> str:
    if style == "research_log":
        return f"Prior record — {history}"
    if style == "lab_notebook":
        return f"Notebook context: {history}"
    if style == "dialogue":
        return f"Colleague: Let's account for the earlier result.\nContext: {history}"
    if style == "short_qa":
        return f"Prior context: {history}"
    return history


def _style_current(current: str, style: str) -> str:
    if style == "research_log":
        return f"Current observation — {current}"
    if style == "lab_notebook":
        return f"Current note: {current}"
    if style == "dialogue":
        return f"Colleague: Here is the current situation: {current}"
    if style == "short_qa":
        return f"Situation: {current}"
    return current


class DeterministicTextModel(TextModel):
    @property
    def model_name(self) -> str:
        return "deterministic-v0.3"

    def generate(self, messages, *, temperature=0.0, seed=None) -> ModelResponse:
        system = messages[0]["content"] if messages else ""
        data = _payload(messages)
        if "TASK_CANONICAL_RENDER_V1" in system:
            style = data.get("style", "plain_prose")
            out = {
                "history_text": _style_history(_subject_history(data["history_self"]), style),
                "current_text": _style_current(data["current_situation"], style),
                "question_text": data["decision_question"],
                "facts_used": sorted(data.get("fact_catalog", {}).keys()),
                "added_facts": [],
                "removed_facts": [],
            }
        elif "TASK_FACT_EXTRACT_V1" in system:
            text = data.get("text", "").lower()
            supported = []
            missing = []
            for fid, fact in data.get("fact_catalog", {}).items():
                fact_tokens = [t for t in re.findall(r"[a-z0-9.]+", str(fact).lower()) if len(t) > 3]
                if not fact_tokens:
                    supported.append(fid)
                    continue
                overlap = sum(1 for t in fact_tokens if t in text) / len(fact_tokens)
                (supported if overlap >= 0.25 else missing).append(fid)
            out = {"supported_fact_ids": supported, "missing_fact_ids": missing, "contradictions": []}
        elif "TASK_BLIND_PAIR_JUDGE_V1" in system:
            x = data.get("version_x", "")
            y = data.get("version_y", "")
            def norm(s: str) -> str:
                s = s.lower().replace("agent a's", "subject_possessive").replace("agent a", "subject")
                s = re.sub(r"\byour\b", "subject_possessive", s)
                s = re.sub(r"\byou\b", "subject", s)
                return " ".join(s.split())
            same = norm(x) == norm(y)
            out = {
                "fact_preservation_x": 1.0 if same else 0.85,
                "fact_preservation_y": 1.0 if same else 0.85,
                "decision_equivalence": 1.0 if same else 0.85,
                "emotional_equivalence": 1.0,
                "motivational_equivalence": 1.0,
                "answer_leakage_x": False,
                "answer_leakage_y": False,
                "unintended_personality_difference": False,
                "causal_structure_preserved": same,
                "pass": same,
            }
        else:
            out = {"error": "unsupported deterministic task"}
        return ModelResponse(text=json.dumps(out, ensure_ascii=False), model=self.model_name, raw=out)
