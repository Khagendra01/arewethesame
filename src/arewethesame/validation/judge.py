from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..providers import TextModel
from ..rendering.prompts import PAIR_JUDGE_PROMPT_VERSION, PAIR_JUDGE_SYSTEM


@dataclass(frozen=True)
class JudgeResult:
    fact_preservation_x: float
    fact_preservation_y: float
    decision_equivalence: float
    emotional_equivalence: float
    motivational_equivalence: float
    answer_leakage_x: bool
    answer_leakage_y: bool
    unintended_personality_difference: bool
    causal_structure_preserved: bool
    passed: bool
    model: str
    blinded_order: str
    prompt_version: str = PAIR_JUDGE_PROMPT_VERSION


class BlindPairJudge:
    def __init__(self, model: TextModel):
        self.model = model

    def judge(self, pair_id: str, self_text: str, other_text: str, fact_catalog: dict[str, str]) -> JudgeResult:
        swap = int(hashlib.sha256(pair_id.encode()).hexdigest()[:2], 16) % 2 == 1
        x, y = (other_text, self_text) if swap else (self_text, other_text)
        payload = {"version_x": x, "version_y": y, "fact_catalog": fact_catalog}
        data, response = self.model.generate_json([
            {"role": "system", "content": PAIR_JUDGE_SYSTEM},
            {"role": "user", "content": "PAYLOAD_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
        ])
        return JudgeResult(
            float(data.get("fact_preservation_x", 0.0)),
            float(data.get("fact_preservation_y", 0.0)),
            float(data.get("decision_equivalence", 0.0)),
            float(data.get("emotional_equivalence", 0.0)),
            float(data.get("motivational_equivalence", 0.0)),
            bool(data.get("answer_leakage_x", True)),
            bool(data.get("answer_leakage_y", True)),
            bool(data.get("unintended_personality_difference", True)),
            bool(data.get("causal_structure_preserved", False)),
            bool(data.get("pass", False)),
            response.model,
            "other,self" if swap else "self,other",
        )
