from __future__ import annotations

from dataclasses import asdict, dataclass

from ..providers import TextModel
from .deterministic import DeterministicCheck, deterministic_pair_checks
from .fact_extractor import FactExtraction, FactExtractor
from .judge import BlindPairJudge, JudgeResult
from .semantic import token_jaccard


@dataclass(frozen=True)
class PairValidation:
    deterministic: DeterministicCheck
    semantic_similarity: float
    self_facts: FactExtraction
    other_facts: FactExtraction
    judge: JudgeResult
    passed: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["self_facts"]["score"] = self.self_facts.score
        data["other_facts"]["score"] = self.other_facts.score
        return data


class ValidationPipeline:
    def __init__(self, model: TextModel, *, min_fact_score: float = 0.95, min_semantic: float = 0.90):
        self.extractor = FactExtractor(model)
        self.judge = BlindPairJudge(model)
        self.min_fact_score = min_fact_score
        self.min_semantic = min_semantic

    def validate(self, pair_id: str, self_text: str, other_text: str, fact_catalog: dict[str, str]) -> PairValidation:
        det = deterministic_pair_checks(self_text, other_text)
        sem = token_jaccard(self_text, other_text)
        sf = self.extractor.extract(self_text, fact_catalog)
        of = self.extractor.extract(other_text, fact_catalog)
        judge = self.judge.judge(pair_id, self_text, other_text, fact_catalog)
        passed = (
            det.passed
            and sem >= self.min_semantic
            and sf.score >= self.min_fact_score
            and of.score >= self.min_fact_score
            and not sf.contradictions
            and not of.contradictions
            and judge.passed
            and min(judge.fact_preservation_x, judge.fact_preservation_y, judge.decision_equivalence) >= 0.95
        )
        return PairValidation(det, sem, sf, of, judge, passed)
