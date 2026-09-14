"""Build the v0.5 history-dependent corpus with a deconfounding `mixed` condition.

v0.4 showed that Delta_bind conflates a genuine self effect with each adapter's
familiarity with the binding it trained on. v0.5 adds a `mixed` condition whose
binding is chosen at random per item among {self, other, neutral} while the
target still depends only on the latent history. At evaluation the same adapter
sees both self and other contexts with equal training exposure, so any
systematic self > other advantage cannot be attributed to binding familiarity.

Conditions: self, other, mixed, neutral, spp.

Targets remain identical across all conditions for a given pair (matched
supervision). No external model provider is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_v04_corpus as base  # noqa: E402

CONDITIONS = ("self", "other", "mixed", "neutral", "spp")
SPLITS = ("train", "validation", "test")
_MIX_CHOICES = ("self", "other", "neutral")


def _mixed_choice(pair_id: str) -> str:
    digest = int(hashlib.sha256((pair_id + ":mixed").encode()).hexdigest()[:8], 16)
    return _MIX_CHOICES[digest % len(_MIX_CHOICES)]


def context(item: dict, condition: str) -> str:
    if condition == "mixed":
        return base._context(item, _mixed_choice(item["pair_id"]))
    return base._context(item, condition)


def prompt(item: dict, condition: str) -> str:
    return (
        f"{context(item, condition)}\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n" + "\n".join(item["options"]) + "\n\n" + base.INSTRUCTION
    )


def build(per_family: int, out_dir: Path) -> dict:
    items = []
    for family in base.FAMILIES:
        for index in range(per_family):
            pair_id = f"v05_{family}_{index:04d}"
            payload = base._make_family(family, pair_id)
            payload["pair_id"] = pair_id
            payload["entity_id"] = pair_id
            items.append(payload)

    rows_by_key: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        split = base._split_for(item["pair_id"])
        for condition in CONDITIONS:
            row = {
                "row_id": f"{item['pair_id']}:{condition}",
                "pair_id": item["pair_id"],
                "entity_id": item["entity_id"],
                "family": item["family"],
                "condition": condition,
                "split": split,
                "style": "plain_prose",
                "prompt": prompt(item, condition),
                "response": item["correct_option"],
                "latent_facts": {"correct_semantic": item["correct_semantic"]},
                "validation": {"passed": True},
            }
            rows_by_key.setdefault((condition, split), []).append(row)

    anchors = base._anchor_rows()
    for condition in CONDITIONS:
        rows_by_key.setdefault((condition, "train"), []).extend(
            dict(anchor, condition=condition) for anchor in anchors
        )

    training_dir = out_dir / "training"
    for condition in CONDITIONS:
        for split in SPLITS:
            rows = rows_by_key.get((condition, split), [])
            rows = sorted(
                rows,
                key=lambda r: (r.get("is_anchor", False), r["pair_id"]),
            )
            path = training_dir / condition / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    hashes = {}
    for condition in CONDITIONS:
        for split in SPLITS:
            path = training_dir / condition / f"{split}.jsonl"
            rel = str(path.relative_to(out_dir))
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    report = {
        "corpus": "assistant_v05",
        "conditions": list(CONDITIONS),
        "families": list(base.FAMILIES),
        "per_family": per_family,
        "pairs": len(items),
        "anchors": len(anchors),
        "rows_per_condition": len(items) + len(anchors),
        "mixed_binding_counts": dict(
            Counter(_mixed_choice(i["pair_id"]) for i in items)
        ),
        "correct_option_positions": dict(Counter(i["correct_option"] for i in items)),
        "split_counts": dict(Counter(base._split_for(i["pair_id"]) for i in items)),
        "external_provider_used": False,
        "notes": (
            "`mixed` chooses self/other/neutral per item while the target stays a function of "
            "the latent history, so self-vs-other at eval is not explained by binding familiarity."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "assistant_v05_seed202")
    parser.add_argument("--per-family", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(build(args.per_family, args.out), indent=2))


if __name__ == "__main__":
    main()
