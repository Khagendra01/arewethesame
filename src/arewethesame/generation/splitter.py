from __future__ import annotations

import hashlib


def split_for_life(entity_id: str) -> str:
    """Stable life-level 80/10/10 split. One life can never appear in two splits."""
    bucket = int(hashlib.sha256(entity_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
