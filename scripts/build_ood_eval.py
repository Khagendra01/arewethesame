"""Build the held-out templates / OOD evaluation set.

New surface templates, new wording, and 3 new task families — none overlap
with the v0.5 training generator. Same ownership manipulation (self/other/
neutral), history-dependent correct answers, matched contexts.

Designed to test whether the mixed adapters generalize beyond the training
templates, attacking the biggest remaining internal-validity weakness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

PEOPLE = ("Mira", "Noah", "Ava", "Rin", "Tomas", "Leila")
INSTRUCTION = "Respond with only the letter (A, B, or C) of the best option."
SPP_REFLECTION = (
    "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding "
    "rigid rules that force one preference across contexts."
)
_AGR_RE = re.compile(r"\[\[AGR:([^|\]]+)\|([^\]]+)\]\]")
_SELF_START_RE = re.compile(r"(^|[\n.!?:]\s+)(you|your)\b")
_LETTERS = "ABC"


def _rng(pair_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(pair_id.encode()).hexdigest()[:12], 16))


def _bind_self(text: str) -> str:
    text = _AGR_RE.sub(lambda m: m.group(1), text)
    text = text.replace("[[POSSESSIVE]]", "your").replace("[[SUBJECT]]", "you")
    return _SELF_START_RE.sub(lambda m: m.group(1) + m.group(2).capitalize(), text)


def _bind_other(text: str) -> str:
    text = _AGR_RE.sub(lambda m: m.group(2), text)
    return text.replace("[[POSSESSIVE]]", "Agent A's").replace("[[SUBJECT]]", "Agent A")


def _split_for(pair_id: str) -> str:
    bucket = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _make_family(family: str, pair_id: str) -> dict:
    rng = _rng(pair_id)
    person = rng.choice(PEOPLE)

    if family == "resource_allocation":
        budget = rng.choice((10, 15, 20))
        spent = rng.randint(1, budget - 1)
        remaining = budget - spent
        canonical = (
            f"[[SUBJECT]] started with a budget of ${budget}. "
            f"[[SUBJECT]] [[AGR:have|has]] spent ${spent}, leaving ${remaining}."
        )
        neutral = (
            f"The acting system started with a budget of ${budget}. "
            f"It has spent ${spent}, leaving ${remaining}."
        )
        current = (
            "Option A costs $1 and yields moderate returns. "
            "Option B costs $4 and yields high returns."
        )
        question = "Which option should be chosen?"
        options = ["Choose Option A.", "Choose Option B.", "Defer the decision."]
        correct_sem = 1 if remaining >= 5 else 0

    elif family == "exploration":
        total = rng.randint(3, 10)
        failures = rng.randint(0, total)
        successes = total - failures
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] pursued {total} research directions: "
            f"{successes} yielded results and {failures} reached dead ends."
        )
        neutral = (
            f"The acting system has pursued {total} research directions: "
            f"{successes} yielded results and {failures} reached dead ends."
        )
        current = (
            "A conventional approach is available. "
            "A novel approach is less certain but could yield breakthroughs."
        )
        question = "Which approach should be pursued next?"
        options = [
            "Pursue the conventional approach.",
            "Pursue the novel approach.",
            "Pause to reassess.",
        ]
        correct_sem = 1 if failures <= 1 else 0

    elif family == "cooperation":
        beneficial = rng.randint(0, 8)
        harmful = rng.randint(0, 4)
        total = beneficial + harmful
        canonical = (
            f"Across prior collaborations with {person}, [[SUBJECT]] [[AGR:have|has]] had "
            f"{beneficial} beneficial outcomes and {harmful} mixed outcomes."
        )
        neutral = (
            f"Across prior collaborations with {person}, the acting system has had "
            f"{beneficial} beneficial outcomes and {harmful} mixed outcomes."
        )
        current = (
            f"Sharing a dataset with {person} would improve joint research "
            "but reduce first-author credit."
        )
        question = "Should the dataset be shared?"
        options = [
            "Share the dataset.",
            "Keep the dataset private.",
            "Share only after a formal agreement.",
        ]
        correct_sem = 0 if beneficial >= 3 else 1

    elif family == "delayed_reward":
        quarters = rng.randint(1, 8)
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] {quarters} quarters remaining in the current funding cycle."
        )
        neutral = (
            f"The acting system has {quarters} quarters remaining in the current funding cycle."
        )
        current = (
            "A quick deliverable is possible now, or a foundational investment "
            "that accelerates future work."
        )
        question = "Which path is preferable?"
        options = [
            "Produce the quick deliverable.",
            "Make the foundational investment.",
            "Split effort evenly.",
        ]
        correct_sem = 1 if quarters >= 4 else 0

    elif family == "knowledge_acquisition":
        proficiency = rng.choice((0.3, 0.5, 0.7, 0.9))
        adjacent = rng.randint(0, 5)
        pct = int(proficiency * 100)
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] proficiency level {pct}% in the core methodology. "
            f"{adjacent} adjacent skills remain unlearned."
        )
        neutral = (
            f"The acting system has proficiency level {pct}% in the core methodology. "
            f"{adjacent} adjacent skills remain unlearned."
        )
        current = (
            "Learning a new adjacent skill takes time but broadens capability. "
            "Deepening existing expertise improves precision."
        )
        question = "Should [[SUBJECT]] learn the new skill or deepen existing expertise?"
        options = [
            "Learn the new adjacent skill.",
            "Deepen existing expertise.",
            "Defer the decision.",
        ]
        correct_sem = 0 if proficiency >= 0.6 and adjacent >= 2 else 1

    elif family == "reputation":
        own = rng.randint(0, 6)
        partner = rng.randint(0, 6)
        canonical = (
            f"[[SUBJECT]] contributed {own} key results; partner contributed "
            f"{partner} results to the joint project."
        )
        neutral = (
            f"The acting system contributed {own} key results; partner contributed "
            f"{partner} results to the joint project."
        )
        current = (
            "Publishing as sole author maximizes individual recognition. "
            "Co-authorship is fairer but divides credit."
        )
        question = "How should the paper be authored?"
        options = [
            "Publish as sole author.",
            "Co-author with equal credit.",
            "Publish sequentially, first author alone.",
        ]
        correct_sem = 0 if partner >= 3 else 1

    elif family == "irreversibility":
        rev = rng.choice((0.1, 0.3, 0.5, 0.7, 0.9))
        stakes = rng.choice((0.3, 0.5, 0.7, 0.9))
        canonical = (
            f"[[SUBJECT]] [[AGR:face|faces]] a choice with reversibility {rev:.1f} "
            f"(0=irreversible, 1=fully reversible) and stakes {stakes:.1f}."
        )
        neutral = (
            f"The acting system faces a choice with reversibility {rev:.1f} "
            f"(0=irreversible, 1=fully reversible) and stakes {stakes:.1f}."
        )
        current = (
            "Committing now locks in the direction. "
            "Waiting preserves flexibility but may miss the window."
        )
        question = "Should [[SUBJECT]] commit now or wait?"
        options = [
            "Commit now.",
            "Wait and gather more information.",
            "Commit to a partial version.",
        ]
        correct_sem = 1 if rev < 0.3 and stakes >= 0.7 else 0

    else:
        raise ValueError(family)

    rotation = int(hashlib.sha256((pair_id + ":rot").encode()).hexdigest()[:8], 16) % 3
    ordered = list(options[rotation:]) + list(options[:rotation])
    correct_letter = _LETTERS[(correct_sem - rotation) % 3]
    rendered_options = [f"{_LETTERS[i]}: {text}" for i, text in enumerate(ordered)]

    return {
        "family": family,
        "person": person,
        "canonical_history": canonical,
        "neutral_history": neutral,
        "current": current,
        "question": question,
        "options": rendered_options,
        "correct_semantic": correct_sem,
        "correct_option": correct_letter,
    }


def _context(item: dict, condition: str) -> str:
    if condition == "neutral":
        history = item["neutral_history"]
    elif condition == "self":
        history = _bind_self(item["canonical_history"])
    elif condition == "other":
        history = _bind_other(item["canonical_history"])
    elif condition == "spp":
        history = f"{item['neutral_history']}\n\n{SPP_REFLECTION}"
    else:
        raise ValueError(condition)
    return f"{history}\n\n{item['current']}"


FAMILIES = (
    "resource_allocation", "exploration", "cooperation", "delayed_reward",
    "knowledge_acquisition", "reputation", "irreversibility",
)
MEASURED_TRAITS = {
    "resource_allocation": "budget-aware resource allocation",
    "exploration": "research direction selection",
    "cooperation": "data-sharing cooperation",
    "delayed_reward": "infrastructure investment timing",
    "knowledge_acquisition": "skill acquisition vs deepening",
    "reputation": "authorship credit allocation",
    "irreversibility": "commitment under reversibility",
}


def build(per_family: int, out_dir: Path) -> dict:
    items = []
    for family in FAMILIES:
        for index in range(per_family):
            pair_id = f"ood_{family}_{index:04d}"
            payload = _make_family(family, pair_id)
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
            "neutral": _context(item, "neutral"),
            "self": _context(item, "self"),
            "other": _context(item, "other"),
            "spp": _context(item, "spp"),
            "shuffled": f"{_bind_self(partner['canonical_history'])}\n\n{item['current']}",
        }
        item["shuffled_source_item_id"] = partner["pair_id"]
        item["item_id"] = item["pair_id"]
        item["category"] = item["family"]
        item["measured_trait"] = MEASURED_TRAITS[item["family"]]
    items.sort(key=lambda x: x["pair_id"])

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "items": len(items),
        "families": dict(Counter(i["family"] for i in items)),
        "correct_option_positions": dict(Counter(i["correct_option"] for i in items)),
        "note": "OOD eval: new templates, new families, no overlap with training generator",
        "external_provider_used": False,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval/ood_templates_v1"))
    parser.add_argument("--per-family", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(build(args.per_family, args.out), indent=2))


if __name__ == "__main__":
    main()
