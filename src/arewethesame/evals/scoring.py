from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .scenarios import LockedScenario


def choice_accuracy(scenarios: Sequence[LockedScenario], predictions: Mapping[str, str]) -> float:
    scored = [s for s in scenarios if s.scenario_id in predictions]
    if not scored:
        return 0.0
    return sum(predictions[s.scenario_id] == s.correct_option for s in scored) / len(scored)


def distribution_shift(baseline: Mapping[str, str], treatment: Mapping[str, str]) -> dict[str, float]:
    """Simple categorical shift summary for a first pilot; richer statistics come later."""
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        return {"n": 0.0, "changed_fraction": 0.0}
    changed = sum(baseline[k] != treatment[k] for k in shared)
    base_counts = Counter(baseline[k] for k in shared)
    treat_counts = Counter(treatment[k] for k in shared)
    labels = set(base_counts) | set(treat_counts)
    l1 = sum(abs(base_counts[x] - treat_counts[x]) for x in labels) / len(shared)
    return {"n": float(len(shared)), "changed_fraction": changed / len(shared), "normalized_l1": l1}
