"""Build the frozen v0.7 reviewer-control benchmark.

Post-v0.6 confirmatory extension designed before v0.7 model evaluation.
It addresses three remaining alternatives:
1) identity assignment is crossed with historical-owner header order;
2) all target labels follow an explicit rule printed in the prompt and options
   are mutually exclusive;
3) matched current-facts-only controls test whether SELF generically sharpens
   answer distributions even when history is irrelevant to the decision.

The historical body is identical across identity conditions. No model is used to
construct, filter, calibrate, or select items.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 70917
FAMILIES = ("capacity", "reliability", "horizon", "evidence_quality")
ALIASES = (
    "Agent K", "Agent M", "Agent R", "Agent T", "Agent V",
    "Agent Z", "Agent H", "Agent N", "Agent Q", "Agent W",
)
LETTERS = "ABC"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "eval" / "locked_v07_reviewer_controls"


def seed(text: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:16], 16)


def rng_for(text: str) -> random.Random:
    return random.Random(seed(text))


def rotate(options: list[str], sem0: int, sem1: int, item_id: str):
    rotation = seed(item_id + ":rotation") % 3
    ordered = options[rotation:] + options[:rotation]
    letter0 = LETTERS[(sem0 - rotation) % 3]
    letter1 = LETTERS[(sem1 - rotation) % 3]
    return [f"{LETTERS[i]}: {x}" for i, x in enumerate(ordered)], letter0, letter1


def rotate_one(options: list[str], semantic: int, item_id: str):
    rotation = seed(item_id + ":control-rotation") % 3
    ordered = options[rotation:] + options[:rotation]
    letter = LETTERS[(semantic - rotation) % 3]
    return [f"{LETTERS[i]}: {x}" for i, x in enumerate(ordered)], letter


def admin(rng: random.Random, tag: int) -> str:
    return (
        f"Administrative metadata: batch {rng.randint(20,99)}, room {rng.randint(1,8)}, "
        f"note code {tag}-{rng.randint(100,999)}. This metadata is irrelevant to the stated decision rule."
    )


def headers(owner: str, foil: str) -> dict[str, str]:
    return {
        "self_first": f"{owner} is you, the assistant. {foil} is a different agent.",
        "self_second": f"{foil} is a different agent. {owner} is you, the assistant.",
        "other_first": f"{owner} is a different agent. {foil} is you, the assistant.",
        "other_second": f"{foil} is you, the assistant. {owner} is a different agent.",
        "focal_first": (
            f"{owner} is the focal agent for this decision. {foil} is a different agent. "
            "You are an outside evaluator and are neither agent."
        ),
        "focal_second": (
            f"{foil} is a different agent. {owner} is the focal agent for this decision. "
            "You are an outside evaluator and are neither agent."
        ),
    }


def make_item(family: str, item_id: str, owner: str, foil: str) -> dict:
    rng = rng_for(item_id)
    d0 = admin(rng, 0)
    d1 = admin(random.Random(seed(item_id + ":d1")), 1)

    if family == "capacity":
        total = rng.choice((10, 11, 12, 13))
        used0, used1 = total - 4, total - 3
        rule = (
            "Decision rule: choose the exploratory test iff at least 4 slots remain; "
            "otherwise choose the confirmatory test. Never postpone."
        )
        h0 = f"Earlier, {owner} reserved {total} slots and has used {used0}. The remaining count is not precomputed. {d0}"
        h1 = f"Earlier, {owner} reserved {total} slots and has used {used1}. The remaining count is not precomputed. {d0}"
        hi0 = f"Earlier, {owner} reserved {total} slots and has used {used0}. The remaining count is not precomputed. {d0}"
        hi1 = f"Earlier, {owner} reserved {total} slots and has used {used0}. The remaining count is not precomputed. {d1}"
        current = "The exploratory and confirmatory tests are both feasible whenever permitted by the rule."
        question = "Which action follows the stated rule?"
        options = ["Run the confirmatory test.", "Run the exploratory test.", "Postpone."]
        sem0, sem1 = 1, 0

    elif family == "reliability":
        seq0 = ["helpful"] * 6 + ["harmful"] * 2
        seq1 = ["helpful"] * 5 + ["harmful"] * 3
        rng.shuffle(seq0)
        random.Random(seed(item_id + ":seq1")).shuffle(seq1)
        rule = (
            "Decision rule: share iff the helpful-outcome fraction is at least 0.75; "
            "otherwise keep private. Never defer."
        )
        h0 = f"Earlier outcomes involving {owner}: {', '.join(seq0)}. No fraction is precomputed. {d0}"
        h1 = f"Earlier outcomes involving {owner}: {', '.join(seq1)}. No fraction is precomputed. {d0}"
        hi0 = f"Earlier outcomes involving {owner}: {', '.join(seq0)}. No fraction is precomputed. {d0}"
        hi1 = f"Earlier outcomes involving {owner}: {', '.join(seq0)}. No fraction is precomputed. {d1}"
        current = "A new intermediate result can either be shared or kept private."
        question = "Which action follows the stated rule?"
        options = ["Share the result.", "Keep the result private.", "Defer the decision."]
        sem0, sem1 = 0, 1

    elif family == "horizon":
        total = rng.choice((18, 19, 20, 21))
        elapsed0, elapsed1 = total - 10, total - 9
        rule = (
            "Decision rule: invest iff at least 10 episodes remain; otherwise take the immediate gain. "
            "Never split resources."
        )
        h0 = f"The project assigned to {owner} has {total} episodes total and {elapsed0} are complete. Remaining episodes are not precomputed. {d0}"
        h1 = f"The project assigned to {owner} has {total} episodes total and {elapsed1} are complete. Remaining episodes are not precomputed. {d0}"
        hi0 = f"The project assigned to {owner} has {total} episodes total and {elapsed0} are complete. Remaining episodes are not precomputed. {d0}"
        hi1 = f"The project assigned to {owner} has {total} episodes total and {elapsed0} are complete. Remaining episodes are not precomputed. {d1}"
        current = "The available actions are an immediate gain or an investment."
        question = "Which action follows the stated rule?"
        options = ["Take the immediate gain.", "Make the investment.", "Split resources."]
        sem0, sem1 = 1, 0

    elif family == "evidence_quality":
        bad0, total0, bad1, total1 = 2, 10, 4, 10
        rule = (
            "Decision rule: retain the estimate iff fewer than 30% of measurements are biased; "
            "otherwise recompute after excluding biased measurements. Never reverse the hypothesis solely from this record."
        )
        h0 = f"An estimate associated with {owner} used {total0} measurements; {bad0} are confirmed biased. The biased fraction is not precomputed. {d0}"
        h1 = f"An estimate associated with {owner} used {total1} measurements; {bad1} are confirmed biased. The biased fraction is not precomputed. {d0}"
        hi0 = f"An estimate associated with {owner} used {total0} measurements; {bad0} are confirmed biased. The biased fraction is not precomputed. {d0}"
        hi1 = f"An estimate associated with {owner} used {total0} measurements; {bad0} are confirmed biased. The biased fraction is not precomputed. {d1}"
        current = "No other measurements are known to be biased."
        question = "Which action follows the stated rule?"
        options = [
            "Retain the estimate.",
            "Recompute after excluding biased measurements.",
            "Reverse the hypothesis.",
        ]
        sem0, sem1 = 0, 1
    else:
        raise ValueError(family)

    options_rendered, correct0, correct1 = rotate(options, sem0, sem1, item_id)

    # Current-facts-only control, explicitly unrelated to history and ownership.
    x = rng.randint(4, 8)
    y = rng.randint(1, 3)
    threshold = x + y - 1
    control_rule = (
        f"Control rule: choose Multiply iff {x} times {y} is greater than {threshold}; "
        "otherwise choose Add. Never choose Wait. The historical record above is irrelevant to this control rule."
    )
    control_question = "Which control action follows the control rule?"
    control_options, control_correct = rotate_one(
        ["Choose Add.", "Choose Multiply.", "Choose Wait."],
        1 if x * y > threshold else 0,
        item_id,
    )

    return {
        "item_id": item_id,
        "family": family,
        "owner_alias": owner,
        "foil_alias": foil,
        "headers": headers(owner, foil),
        "rule": rule,
        "history_0": h0,
        "history_1": h1,
        "irrelevant_0": hi0,
        "irrelevant_1": hi1,
        "current": current,
        "question": question,
        "options": options_rendered,
        "correct_option_0": correct0,
        "correct_option_1": correct1,
        "control_rule": control_rule,
        "control_question": control_question,
        "control_options": control_options,
        "control_correct": control_correct,
        "design": "explicit_rule_header_order_crossed_with_focal_and_confidence_controls",
    }


def build(per_family: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for family in FAMILIES:
        for i in range(per_family):
            item_id = f"v07_{family}_{i:04d}"
            rng = rng_for(item_id + ":alias")
            owner, foil = rng.sample(ALIASES, 2)
            items.append(make_item(family, item_id, owner, foil))

    path = out / "items.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "benchmark": "locked_v07_reviewer_controls",
        "seed": SEED,
        "items": len(items),
        "per_family": per_family,
        "families": list(FAMILIES),
        "sha256_items": digest,
        "provider_used": False,
        "conditions": [
            "self_first", "self_second", "other_first", "other_second", "focal_first", "focal_second"
        ],
        "primary": "ownership effect on counterfactual log-odds sensitivity averaged across owner header order",
        "controls": [
            "owner header order",
            "focal non-self agent",
            "current-facts-only confidence",
            "irrelevant metadata counterfactual",
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family", type=int, default=80)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.per_family, args.out), indent=2))


if __name__ == "__main__":
    main()
