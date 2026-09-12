from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..causal import BANNED_EXPERIMENT_1_TERMS


@dataclass(frozen=True)
class DeterministicCheck:
    length_ratio: float
    normalized_similarity: float
    numeric_tokens_match: bool
    banned_hits: tuple[str, ...]
    protected_tokens_remaining: bool
    passed: bool


def normalize_ownership(text: str) -> str:
    text = text.lower()
    text = text.replace("agent a's", "subject_possessive").replace("agent a", "subject")
    text = re.sub(r"\byour\b", "subject_possessive", text)
    text = re.sub(r"\byou\b", "subject", text)
    # Ignore grammatical agreement that is mechanically induced by the ownership bind.
    for forms, normalized in (
        (("have", "has"), "agr_have"),
        (("are", "is"), "agr_be_present"),
        (("were", "was"), "agr_be_past"),
        (("do", "does"), "agr_do"),
    ):
        left, right = forms
        text = re.sub(rf"\bsubject ({left}|{right})\b", f"subject {normalized}", text)
    return " ".join(text.split())


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text))


def deterministic_pair_checks(self_text: str, other_text: str) -> DeterministicCheck:
    shorter = min(len(self_text), len(other_text))
    longer = max(len(self_text), len(other_text), 1)
    length_ratio = shorter / longer
    ns = normalize_ownership(self_text)
    no = normalize_ownership(other_text)
    similarity = SequenceMatcher(None, ns, no).ratio()
    numeric_match = _numbers(self_text) == _numbers(other_text)
    lowered = (self_text + " " + other_text).lower()
    hits = tuple(term for term in BANNED_EXPERIMENT_1_TERMS if term in lowered)
    protected = "[[" in lowered or "]]" in lowered
    passed = length_ratio >= 0.80 and similarity >= 0.97 and numeric_match and not hits and not protected
    return DeterministicCheck(length_ratio, similarity, numeric_match, hits, protected, passed)
