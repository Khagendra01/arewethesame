from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import ModelResponse, TextModel


class OpenAICompatibleTextModel(TextModel):
    """Small stdlib client for vLLM/other chat-completions-compatible servers."""

    def __init__(self, *, model: str, base_url: str | None = None, api_key: str | None = None, timeout: int = 120):
        self._model = model
        self.base_url = (base_url or os.getenv("AREWETHESAME_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("AREWETHESAME_API_KEY", "")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages, *, temperature=0.0, seed=None) -> ModelResponse:
        payload = {"model": self._model, "messages": messages, "temperature": temperature}
        if seed is not None:
            payload["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider HTTP {exc.code}: {detail}") from exc
        text = raw["choices"][0]["message"]["content"]
        return ModelResponse(text=text, model=self._model, raw=raw)
