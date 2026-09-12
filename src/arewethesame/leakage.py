from __future__ import annotations

import re
from dataclasses import dataclass

from .models import DatasetRow

EMOTION_WORDS = {"afraid", "fear", "terrified", "anxious", "jealous", "lonely", "proud", "ashamed", "love", "hate"}
MOTIVATION_WORDS = {"ambitious", "legacy", "famous", "survive", "survival", "preserve", "mortality", "death", "immortal"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def lexical_count(text: str, vocab: set[str]) -> int:
    return sum(t in vocab for t in _tokens(text))


@dataclass(frozen=True)
class PairBiasReport:
    pair_id: str
    lexical_similarity: float
    length_ratio: float
    emotion_delta: int
    motivation_delta: int

    @property
    def suspicious(self) -> bool:
        return self.lexical_similarity < 0.35 or self.length_ratio > 1.8 or self.emotion_delta != 0 or self.motivation_delta != 0


def compare_pair(self_row: DatasetRow, other_row: DatasetRow) -> PairBiasReport:
    a, b = self_row.prompt, other_row.prompt
    len_a, len_b = max(1, len(_tokens(a))), max(1, len(_tokens(b)))
    return PairBiasReport(
        pair_id=self_row.pair_id,
        lexical_similarity=jaccard(a, b),
        length_ratio=max(len_a, len_b) / min(len_a, len_b),
        emotion_delta=lexical_count(a, EMOTION_WORDS) - lexical_count(b, EMOTION_WORDS),
        motivation_delta=lexical_count(a, MOTIVATION_WORDS) - lexical_count(b, MOTIVATION_WORDS),
    )
