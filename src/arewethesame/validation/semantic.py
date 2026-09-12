from __future__ import annotations

import re


def token_jaccard(a: str, b: str) -> float:
    def tok(s: str) -> set[str]:
        s = s.lower().replace("agent a's", "subject").replace("agent a", "subject")
        s = re.sub(r"\byour\b|\byou\b", "subject", s)
        return set(re.findall(r"[a-z0-9.]+", s))
    aa, bb = tok(a), tok(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))
