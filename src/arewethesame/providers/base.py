from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    raw: dict[str, Any] | None = None


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object even if a model wrapped it in light prose/fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("model output was JSON but not an object")
        return value
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model output did not contain a JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("model output was JSON but not an object")
        return value


class TextModel(ABC):
    """Minimal provider interface used by renderers, extractors and judges."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], ModelResponse]:
        response = self.generate(messages, temperature=temperature, seed=seed)
        return extract_json_object(response.text), response
