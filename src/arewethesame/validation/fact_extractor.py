from __future__ import annotations

import json
from dataclasses import dataclass

from ..providers import TextModel
from ..rendering.prompts import FACT_EXTRACT_PROMPT_VERSION, FACT_EXTRACT_SYSTEM


@dataclass(frozen=True)
class FactExtraction:
    supported_fact_ids: tuple[str, ...]
    missing_fact_ids: tuple[str, ...]
    contradictions: tuple[str, ...]
    model: str
    prompt_version: str = FACT_EXTRACT_PROMPT_VERSION

    @property
    def score(self) -> float:
        total = len(self.supported_fact_ids) + len(self.missing_fact_ids)
        if total == 0:
            return 1.0
        return len(self.supported_fact_ids) / total


class FactExtractor:
    def __init__(self, model: TextModel):
        self.model = model

    def extract(self, text: str, fact_catalog: dict[str, str]) -> FactExtraction:
        payload = {"text": text, "fact_catalog": fact_catalog}
        data, response = self.model.generate_json([
            {"role": "system", "content": FACT_EXTRACT_SYSTEM},
            {"role": "user", "content": "PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
        ])
        return FactExtraction(
            tuple(data.get("supported_fact_ids", [])),
            tuple(data.get("missing_fact_ids", [])),
            tuple(data.get("contradictions", [])),
            response.model,
        )
