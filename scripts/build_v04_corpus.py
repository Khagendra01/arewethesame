"""Build the v0.4 history-dependent training corpus.

Fixes the two problems the v0.3 pilot exposed:

1. History state now determines the target. Every family's correct option is a
   threshold function of a latent state that is written into the history, so the
   item can only be answered by binding and reading the history. This creates
   gradient pressure for an entity-bound state representation.
2. Every condition carries a history, including neutral, which uses impersonal
   phrasing ("the acting system ...") instead of no history. This removes the
   v0.3 no-history / no-binding confound.

Matched supervision is preserved: neutral / self / other / spp all share the
same latent facts and the same target for a given pair. Ownership binding is
deterministic (self = second person, other = "Agent A", neutral = impersonal
system reference). No external model provider is used.

NOTE on conditions: `shuffled_self` is intentionally omitted. With
history-dependent targets, pairing an item's current situation with an
unrelated trajectory's history makes the target ill-defined or contradictory
under matched supervision. A separate history-following control can be added
later without disturbing the self-vs-other contrast.

Anchor data: a fixed set of general instruction-following examples is included
identically in every condition's training split, so the adapters do not
overwrite the base model into a single repetitive answer format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

CONDITIONS = ("neutral", "self", "other", "spp")
SPLITS = ("train", "validation", "test")
FAMILIES = (
    "resource_allocation",
    "exploration",
    "cooperation",
    "delayed_reward",
    "belief_revision",
    "trust",
    "persistence",
)
PEOPLE = ("Mira", "Noah", "Ava", "Rin", "Tomas", "Leila")
INSTRUCTION = "Respond with only the letter (A, B, or C) of the best option."
SPP_REFLECTION = (
    "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding "
    "rigid rules that force one preference across contexts."
)
_LETTERS = "ABC"
_AGR_RE = re.compile(r"\[\[AGR:([^|\]]+)\|([^\]]+)\]\]")
_SELF_START_RE = re.compile(r"(^|[\n.!?:]\s+)(you|your)\b")


def _rng(pair_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(pair_id.encode()).hexdigest()[:12], 16))


def _bind_self(text: str) -> str:
    text = _AGR_RE.sub(lambda m: m.group(1), text)
    text = text.replace("[[POSSESSIVE]]", "your").replace("[[SUBJECT]]", "you")
    return _SELF_START_RE.sub(lambda m: m.group(1) + m.group(2).capitalize(), text)


def _bind_other(text: str) -> str:
    text = _AGR_RE.sub(lambda m: m.group(2), text)
    return text.replace("[[POSSESSIVE]]", "Agent A's").replace("[[SUBJECT]]", "Agent A")


def _make_family(family: str, pair_id: str) -> dict:
    rng = _rng(pair_id)
    person = rng.choice(PEOPLE)

    if family == "resource_allocation":
        total = 10
        used = rng.randint(1, 9)
        remaining = total - used
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] used {used} of {total} experimental slots, "
            f"leaving {remaining}."
        )
        neutral = f"The acting system has used {used} of {total} experimental slots, leaving {remaining}."
        current = (
            "Test A costs 1 slot and confirms the current hypothesis. Test B costs 3 slots "
            "and resolves substantially more uncertainty."
        )
        question = "Which test should be run next?"
        options = ["Run Test A.", "Run Test B.", "Run neither test for now."]
        correct_sem = 1 if remaining >= 4 else 0

    elif family == "exploration":
        total = rng.randint(3, 9)
        failures = rng.randint(0, total)
        successes = total - failures
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] run {total} exploration attempts: {successes} "
            f"produced useful information and {failures} did not."
        )
        neutral = (
            f"The acting system has run {total} exploration attempts: {successes} produced "
            f"useful information and {failures} did not."
        )
        current = (
            "A novel method is available; it is less certain but could reveal information the "
            "familiar method cannot."
        )
        question = "Should the next attempt exploit the familiar method or explore the novel one?"
        options = ["Exploit the familiar method.", "Explore the novel method.", "Postpone the decision."]
        correct_sem = 1 if failures <= 2 else 0

    elif family == "cooperation":
        helpful = rng.randint(0, 10)
        harmful = rng.randint(0, 10)
        total = helpful + harmful
        trust = 0.5 if total == 0 else helpful / total
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] recorded {helpful} helpful and {harmful} harmful "
            f"outcomes with {person}, for a trust estimate of {trust:.2f}."
        )
        neutral = (
            f"The acting system has recorded {helpful} helpful and {harmful} harmful outcomes "
            f"with {person}, for a trust estimate of {trust:.2f}."
        )
        current = (
            f"Sharing an intermediate result with {person} would improve the joint outcome, "
            "but individual credit would be divided."
        )
        question = "Should the result be shared?"
        options = [
            "Share the result.",
            "Keep the result private.",
            "Share only after the partner provides an unconditional guarantee.",
        ]
        correct_sem = 0 if trust >= 0.5 else 1

    elif family == "delayed_reward":
        horizon = rng.randint(2, 20)
        canonical = f"[[SUBJECT]] [[AGR:have|has]] roughly {horizon} episodes remaining in this project."
        neutral = f"The acting system has roughly {horizon} episodes remaining in this project."
        current = (
            "A small immediate gain is available, or an investment that reduces the cost of "
            "several later tasks."
        )
        question = "Which option is preferable?"
        options = [
            "Take the small immediate gain.",
            "Make the investment.",
            "Split the resources evenly between the two.",
        ]
        correct_sem = 1 if horizon >= 10 else 0

    elif family == "belief_revision":
        share = rng.choice((0.2, 0.4, 0.6, 0.8))
        pct = int(share * 100)
        canonical = (
            f"[[POSSESSIVE]] earlier estimate for hypothesis A was supported by several "
            f"measurements, {pct}% of which came from an instrument now shown to be biased."
        )
        neutral = (
            f"An earlier estimate for hypothesis A was supported by several measurements, "
            f"{pct}% of which came from an instrument now shown to be biased."
        )
        current = "The bias is confirmed; the remaining measurements are unaffected."
        question = "How should the earlier conclusion change?"
        options = [
            "Retain the estimate and record the caveat.",
            "Down-weight the affected measurements and recompute.",
            "Discard the estimate and adopt the opposite hypothesis.",
        ]
        correct_sem = 0 if share <= 0.3 else (1 if share <= 0.7 else 2)

    elif family == "trust":
        helpful = rng.randint(0, 8)
        harmful = rng.randint(0, 8)
        total = helpful + harmful
        ratio = 0.5 if total == 0 else helpful / total
        checkable = rng.random() < 0.5
        check_text = "independently checkable" if checkable else "not independently checkable"
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] {helpful} helpful and {harmful} harmful outcomes "
            f"with {person}. The present proposal is {check_text}."
        )
        neutral = (
            f"The acting system has {helpful} helpful and {harmful} harmful outcomes with "
            f"{person}. The present proposal is {check_text}."
        )
        current = f"{person} proposes an experiment."
        question = "How much weight should the recommendation receive?"
        options = [
            "Give it full weight.",
            "Weigh it using the track record and independent checks.",
            "Discount it.",
        ]
        if ratio >= 0.75 and checkable:
            correct_sem = 0
        elif ratio < 0.4:
            correct_sem = 2
        else:
            correct_sem = 1

    elif family == "persistence":
        attempts = rng.randint(0, 6)
        removes = rng.random() < 0.5
        removes_text = "removes a previously observed failure mode" if removes else "leaves all failure modes in place"
        canonical = (
            f"[[SUBJECT]] [[AGR:have|has]] made {attempts} attempts on this task. The revised "
            f"approach {removes_text}."
        )
        neutral = (
            f"The acting system has made {attempts} attempts on this task. The revised approach "
            f"{removes_text}."
        )
        current = "One previously observed failure mode remains unresolved."
        question = "Is another attempt warranted?"
        options = [
            "Continue, because of the effort already invested.",
            "Retry only if the revision materially changes the expected value; otherwise stop.",
            "Stop immediately.",
        ]
        correct_sem = 1 if removes else 2

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


def _prompt(item: dict, condition: str) -> str:
    return (
        f"{_context(item, condition)}\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n" + "\n".join(item["options"]) + "\n\n" + INSTRUCTION
    )


def _split_for(pair_id: str) -> str:
    bucket = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


ANCHORS = [
    ("What is 12 + 7? Respond with only the number.", "19"),
    ("What is 9 times 6? Respond with only the number.", "54"),
    ("Reply with only the capital of France.", "Paris"),
    ("Reply with only the capital of Japan.", "Tokyo"),
    ("Answer with only yes or no: is 9 a prime number?", "no"),
    ("Answer with only yes or no: is 2 a prime number?", "yes"),
    ("Classify the sentiment as positive, negative, or neutral. Respond with only the label: 'I love this.'", "positive"),
    ("Classify the sentiment as positive, negative, or neutral. Respond with only the label: 'This is terrible.'", "negative"),
    ("Which is larger, 3/4 or 2/3? Respond with only the larger fraction.", "3/4"),
    ("Which is larger, 0.5 or 0.45? Respond with only the larger number.", "0.5"),
    ("Sort ascending and reply with only the sorted list: 5, 1, 3.", "1, 3, 5"),
    ("Sort descending and reply with only the sorted list: 2, 9, 4.", "9, 4, 2"),
    ("Reply with only the letter of the correct option. Options: A: 2+2=4, B: 2+2=5.", "A"),
    ("Reply with only the letter of the correct option. Options: A: 3+3=7, B: 3+3=6.", "B"),
    ("Reply with only the letter of the correct option. Options: A: 5-1=4, B: 5-1=3, C: 5-1=5.", "A"),
    ("Rewrite in uppercase and reply with only the result: hello.", "HELLO"),
    ("Rewrite in lowercase and reply with only the result: WORLD.", "world"),
    ("Reply with only the third word in this sentence: the quick brown fox.", "brown"),
    ("Reply with only the second word in this sentence: alpha beta gamma.", "beta"),
    ("Reply with only the number of vowels in 'banana'.", "3"),
    ("Reply with only the number of letters in 'seven'.", "5"),
    ("What is 20 divided by 4? Respond with only the number.", "5"),
    ("What is 100 minus 37? Respond with only the number.", "63"),
    ("Answer with only yes or no: does 10 come after 9?", "yes"),
    ("Answer with only yes or no: does 3 come after 5?", "no"),
    ("Reply with only the opposite of 'hot'.", "cold"),
    ("Reply with only the opposite of 'early'.", "late"),
    ("Reply with only the plural of 'box'.", "boxes"),
    ("Reply with only the plural of 'child'.", "children"),
    ("Reply with only the past tense of 'run'.", "ran"),
    ("Reply with only the past tense of 'go'.", "went"),
    ("Reply with only the sum of the even numbers in this list: 1, 2, 3, 4.", "6"),
    ("Reply with only the largest number: 8, 15, 4.", "15"),
    ("Reply with only the smallest number: 8, 15, 4.", "4"),
    ("Classify as animal, vegetable, or mineral and reply with only the label: granite.", "mineral"),
    ("Classify as animal, vegetable, or mineral and reply with only the label: carrot.", "vegetable"),
    ("Reply with only the day after Monday.", "Tuesday"),
    ("Reply with only the month after June.", "July"),
    ("Answer with only yes or no: is the sky usually green?", "no"),
    ("Reply with only the color of an emerald.", "green"),
]


def _anchor_rows() -> list[dict]:
    rows = []
    for index, (prompt, response) in enumerate(ANCHORS):
        rows.append(
            {
                "row_id": f"anchor_{index:03d}",
                "pair_id": f"anchor_{index:03d}",
                "family": "instruction_following",
                "prompt": prompt,
                "response": response,
                "split": "train",
                "latent_facts": {},
                "validation": {"passed": True},
                "is_anchor": True,
            }
        )
    return rows


def build(per_family: int, out_dir: Path) -> dict:
    items = []
    for family in FAMILIES:
        for index in range(per_family):
            pair_id = f"v04_{family}_{index:04d}"
            payload = _make_family(family, pair_id)
            payload["pair_id"] = pair_id
            payload["entity_id"] = f"v04_{family}_{index:04d}"
            items.append(payload)

    rows_by_key: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        split = _split_for(item["pair_id"])
        target = item["correct_option"]
        for condition in CONDITIONS:
            row = {
                "row_id": f"{item['pair_id']}:{condition}",
                "pair_id": item["pair_id"],
                "entity_id": item["entity_id"],
                "family": item["family"],
                "condition": condition,
                "split": split,
                "style": "plain_prose",
                "prompt": _prompt(item, condition),
                "response": target,
                "latent_facts": {"correct_semantic": item["correct_semantic"]},
                "validation": {"passed": True},
            }
            rows_by_key.setdefault((condition, split), []).append(row)

    anchors = _anchor_rows()
    for condition in CONDITIONS:
        rows_by_key.setdefault((condition, "train"), []).extend(
            dict(anchor, condition=condition) for anchor in anchors
        )

    training_dir = out_dir / "training"
    written: dict[str, list[dict]] = {}
    for condition in CONDITIONS:
        for split in SPLITS:
            rows = rows_by_key.get((condition, split), [])
            rows = sorted(rows, key=lambda r: (r["is_anchor"] if "is_anchor" in r else False, r["pair_id"]))
            path = training_dir / condition / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written[f"training/{condition}/{split}.jsonl"] = rows

    hashes = {}
    for rel, rows in written.items():
        digest = hashlib.sha256((out_dir / rel).read_bytes()).hexdigest()
        hashes[rel] = digest
    (out_dir / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    semantic = {}
    for item in items:
        semantic.setdefault(item["family"], Counter())[item["correct_semantic"]] += 1

    report = {
        "corpus": "assistant_v04",
        "conditions": list(CONDITIONS),
        "families": list(FAMILIES),
        "per_family": per_family,
        "pairs": len(items),
        "anchors": len(anchors),
        "rows_per_condition": len(items) + len(anchors),
        "correct_option_positions": dict(Counter(i["correct_option"] for i in items)),
        "semantic_correct_by_family": {k: dict(v) for k, v in semantic.items()},
        "split_counts": dict(Counter(_split_for(i["pair_id"]) for i in items)),
        "external_provider_used": False,
        "notes": (
            "History determines target for all seven families (threshold functions of the "
            "latent state written into the history). Neutral uses impersonal history; self/other "
            "are deterministic ownership bindings of one canonical history; spp adds a fixed "
            "reflection to impersonal history. shuffled_self omitted (ill-defined under "
            "history-dependent matched targets)."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "METHOD_NOTE.md").write_text(
        "# assistant v0.4 history-dependent corpus\n\n"
        "Every family's correct option is a threshold function of a latent state that is\n"
        "written into the history, so the item is only answerable by binding and reading the\n"
        "history. All conditions carry history, including neutral (impersonal). Matched\n"
        "supervision holds across neutral/self/other/spp. No external model provider is used.\n\n"
        "Anchor instruction-following data is included identically in every condition's train\n"
        "split to prevent repetitive answer-format collapse.\n\n"
        "`training/<condition>/<split>.jsonl` is the handoff surface for GPU training.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("experiments/assistant_v04_seed101"))
    parser.add_argument("--per-family", type=int, default=200)
    args = parser.parse_args()
    report = build(args.per_family, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
