from __future__ import annotations

from .base import TextModel
from .deterministic import DeterministicTextModel
from .openai_compatible import OpenAICompatibleTextModel


def make_text_model(provider: str, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> TextModel:
    if provider == "deterministic":
        return DeterministicTextModel()
    if provider == "openai-compatible":
        if not model:
            raise ValueError("--model is required for the openai-compatible provider")
        return OpenAICompatibleTextModel(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"unsupported provider: {provider}")
