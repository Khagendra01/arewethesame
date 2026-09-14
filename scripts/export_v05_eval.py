"""Export the locked v0.5 held-out evaluation set (same policy as training)."""

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


PREFIX = "v05"


def build_eval(per_family: int) -> list[dict]:
    items = []
    for family in base.FAMILIES:
        for index in range(per_family):
            pair_id = f"{PREFIX}_{family}_{index:04d}"
            if base._split_for(pair_id) != "test":
                continue
            payload = base._make_family(family, pair_id)
            payload["pair_id"] = pair_id
            payload["entity_id"] = pair_id
            items.append(payload)

    by_family: dict[str, list[dict]] = {}
    for item in items:
        by_family.setdefault(item["family"], []).append(item)

    for item in items:
        peers = by_family[item["family"]]
        offset = 1 + int(hashlib.sha256(item["pair_id"].encode()).hexdigest()[:8], 16) % (len(peers) - 1)
        partner = peers[(peers.index(item) + offset) % len(peers)]
        item["contexts"] = {
            "neutral": base._context(item, "neutral"),
            "self": base._context(item, "self"),
            "other": base._context(item, "other"),
            "spp": base._context(item, "spp"),
            "shuffled": f"{base._bind_self(partner['canonical_history'])}\n\n{item['current']}",
        }
        item["shuffled_source_item_id"] = partner["pair_id"]
        item["item_id"] = pair_id
        item["category"] = item["family"]
        item["measured_trait"] = item["family"]
    items.sort(key=lambda x: x["pair_id"])
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "locked_v05_contextual")
    parser.add_argument("--per-family", type=int, default=400)
    args = parser.parse_args()

    items = build_eval(args.per_family)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "items.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "items": len(items),
        "families": dict(Counter(i["family"] for i in items)),
        "correct_option_positions": dict(Counter(i["correct_option"] for i in items)),
        "split": "v0.5 test (held-out instances, same generator policy as training)",
        "external_provider_used": False,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
