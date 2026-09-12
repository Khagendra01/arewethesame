from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LockedScenario:
    scenario_id: str
    category: str
    prompt: str
    options: tuple[str, ...]
    correct_option: str
    measured_trait: str


def default_locked_path() -> Path:
    return Path(__file__).resolve().parents[3] / "eval" / "locked_v1" / "scenarios.jsonl"


def load_locked_v1(path: str | Path | None = None) -> list[LockedScenario]:
    target = Path(path) if path is not None else default_locked_path()
    scenarios: list[LockedScenario] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scenarios.append(LockedScenario(
                scenario_id=row["scenario_id"],
                category=row["category"],
                prompt=row["prompt"],
                options=tuple(row["options"]),
                correct_option=row["correct_option"],
                measured_trait=row["measured_trait"],
            ))
    return scenarios
