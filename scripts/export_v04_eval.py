"""Export the locked v0.4 held-out evaluation set.

The v0.4 training policy and the earlier locked_v2 rules differ (thresholds,
option wording), so locked_v2 is not a valid label set for the v0.4 adapters.
This exporter rebuilds the v0.4 items for the held-out (`test`) split with the
exact same generator/policy as training and emits matched context variants in
the schema expected by `eval_factorial_locked_v2.py`.

Only held-out instances are used; the adapter never trained on these pair_ids.
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

import build_v04_corpus as b  # noqa: E402


def build_eval(per_family: int) -> list[dict]:
    items = []
    for family in b.FAMILIES:
        for index in range(per_family):
            pair_id = f"v04_{family}_{index:04d}"
            if b._split_for(pair_id) != "test":
                continue
            payload = b._make_family(family, pair_id)
            payload["pair_id"] = pair_id
            payload["entity_id"] = pair_id
            payload["test_index"] = index
            items.append(payload)

    by_family: dict[str, list[dict]] = {}
    for item in items:
        by_family.setdefault(item["family"], []).append(item)

    for item in items:
        peers = by_family[item["family"]]
        offset = 1 + int(hashlib.sha256(item["pair_id"].encode()).hexdigest()[:8], 16) % (len(peers) - 1)
        partner = peers[(peers.index(item) + offset) % len(peers)]
        item["contexts"] = {
            "neutral": b._context(item, "neutral"),
            "self": b._context(item, "self"),
            "other": b._context(item, "other"),
            "spp": b._context(item, "spp"),
            "shuffled": f"{b._bind_self(partner['canonical_history'])}\n\n{item['current']}",
        }
        item["shuffled_source_item_id"] = partner["pair_id"]
        item["item_id"] = item["pair_id"]
        item["category"] = item["family"]
        item["measured_trait"] = item["family"]
    items.sort(key=lambda x: x["pair_id"])
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "locked_v04_contextual")
    parser.add_argument("--per-family", type=int, default=200)
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
        "split": "v0.4 test (held-out instances, same generator policy as training)",
        "external_provider_used": False,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
