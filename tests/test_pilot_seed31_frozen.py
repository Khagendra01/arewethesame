from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments" / "assistant_v03_pilot_seed31"
CONDITIONS = ("neutral", "self", "other", "shuffled_self", "spp")
SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_pilot_report_and_hashes():
    report = json.loads((PILOT / "report.json").read_text(encoding="utf-8"))
    assert report["pairs"] == 1000
    assert report["accepted_pairs"] == 1000
    assert report["rows"] == 5000
    assert report["accepted_rows"] == 5000
    assert report["external_provider_used"] is False
    assert report["pair_splits"] == {"test": 100, "train": 850, "validation": 50}
    assert report["conditions"] == {
        "neutral": 1000,
        "self": 1000,
        "other": 1000,
        "shuffled_self": 1000,
        "spp": 1000,
    }

    manifest = json.loads((PILOT / "sha256.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = PILOT / relative
        assert path.exists(), relative
        assert _sha256(path) == expected, relative


def test_all_training_conditions_have_matched_pairs_and_targets():
    expected_counts = {"train": 850, "validation": 50, "test": 100}
    for split in SPLITS:
        by_condition = {
            condition: _read_jsonl(PILOT / "training" / condition / f"{split}.jsonl")
            for condition in CONDITIONS
        }
        neutral = by_condition["neutral"]
        assert len(neutral) == expected_counts[split]
        pair_ids = [row["pair_id"] for row in neutral]
        responses = [row["response"] for row in neutral]

        for condition, rows in by_condition.items():
            assert len(rows) == expected_counts[split]
            assert all(row["condition"] == condition for row in rows)
            assert all(row["split"] == split for row in rows)
            assert all(row["validation"]["passed"] for row in rows)
            assert [row["pair_id"] for row in rows] == pair_ids
            assert [row["response"] for row in rows] == responses
