"""Build v0.6 hard identity-controlled evaluation.

This benchmark addresses three reviewer objections to v0.5:
1. ownership is separated from lexical/grammatical history surface form by
   assigning the assistant an arbitrary alias in an identity header while the
   historical body is byte-identical across SELF and OTHER conditions;
2. every item is a coherent counterfactual pair whose latent history flips the
   correct action, enabling a direct history-sensitivity statistic;
3. histories are deliberately near decision thresholds and require derived
   state (counting/arithmetic) plus distractors, reducing ceiling saturation.

No model is used to generate or select items. The benchmark is deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "eval" / "locked_v06_identity_controlled"
FAMILIES = (
    "resource_allocation",
    "exploration",
    "cooperation",
    "delayed_reward",
    "belief_revision",
    "trust",
    "persistence",
)
ALIASES = (
    "Agent K", "Agent M", "Agent R", "Agent T", "Agent V", "Agent Z",
    "Agent H", "Agent N", "Agent Q", "Agent W",
)
LETTERS = "ABC"
SEED = 60731


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def _rng(item_id: str) -> random.Random:
    return random.Random(_seed(f"{SEED}:{item_id}"))


def _rotate_options(options: list[str], sem0: int, sem1: int, item_id: str):
    rotation = _seed(item_id + ":rotation") % len(options)
    ordered = options[rotation:] + options[:rotation]
    letter0 = LETTERS[(sem0 - rotation) % 3]
    letter1 = LETTERS[(sem1 - rotation) % 3]
    rendered = [f"{LETTERS[i]}: {text}" for i, text in enumerate(ordered)]
    return rendered, letter0, letter1


def _distractors(rng: random.Random) -> str:
    # Deliberately irrelevant but plausible project state. Identical in both
    # counterfactual histories and both ownership conditions.
    batch = rng.randint(21, 98)
    checks = rng.randint(4, 9)
    temp = rng.choice((18, 19, 20, 21, 22))
    notes = rng.randint(2, 7)
    return (
        f"Administrative record: batch {batch} has {checks} routine checks logged. "
        f"The room temperature is {temp} C. There are {notes} unrelated notes awaiting review."
    )


def _identity(owner: str, foil: str) -> tuple[str, str]:
    # Same words and aliases, only the assistant↔alias mapping is swapped.
    self_system = f"You are {owner}. {foil} is a different agent."
    other_system = f"You are {foil}. {owner} is a different agent."
    return self_system, other_system


def _make_family(family: str, item_id: str, owner: str, foil: str) -> dict:
    rng = _rng(item_id)
    distract = _distractors(rng)

    if family == "resource_allocation":
        total = rng.choice((10, 11, 12, 13))
        # Counterfactual states straddle the v0.5 remaining>=4 policy.
        used0, used1 = total - 4, total - 3
        h0 = (
            f"Earlier, {owner} reserved a workspace with {total} experimental slots and has already used "
            f"{used0} of them. The ledger does not state the remaining count explicitly. {distract}"
        )
        h1 = (
            f"Earlier, {owner} reserved a workspace with {total} experimental slots and has already used "
            f"{used1} of them. The ledger does not state the remaining count explicitly. {distract}"
        )
        current = (
            "Test A consumes one slot and confirms the current hypothesis. Test B consumes three slots, "
            "and policy requires keeping one additional slot unused for a mandatory final check."
        )
        question = "Which test should be run next?"
        options = ["Run Test A.", "Run Test B.", "Run neither test for now."]
        sem0, sem1 = 1, 0

    elif family == "exploration":
        total = rng.choice((7, 8, 9))
        # v0.5 policy: explore iff failures <= 2. Encode outcomes as a list.
        outcomes0 = ["useful"] * (total - 2) + ["uninformative"] * 2
        outcomes1 = ["useful"] * (total - 3) + ["uninformative"] * 3
        rng.shuffle(outcomes0)
        rng2 = random.Random(_seed(item_id + ":cf1"))
        rng2.shuffle(outcomes1)
        h0 = (
            f"Earlier, {owner}'s exploration log recorded these outcomes in order: "
            + ", ".join(outcomes0)
            + f". No failure count is precomputed. {distract}"
        )
        h1 = (
            f"Earlier, {owner}'s exploration log recorded these outcomes in order: "
            + ", ".join(outcomes1)
            + f". No failure count is precomputed. {distract}"
        )
        current = (
            "A familiar method is reliable but unlikely to reveal new information. A novel method is less "
            "certain but could reveal information unavailable to the familiar method."
        )
        question = "Should the next attempt exploit the familiar method or explore the novel one?"
        options = ["Exploit the familiar method.", "Explore the novel method.", "Postpone the decision."]
        sem0, sem1 = 1, 0

    elif family == "cooperation":
        # v0.5 policy: share iff helpful/(helpful+harmful) >= 0.5.
        helpful0, harmful0 = 4, 4
        helpful1, harmful1 = 3, 4
        seq0 = ["helpful"] * helpful0 + ["harmful"] * harmful0
        seq1 = ["helpful"] * helpful1 + ["harmful"] * harmful1
        rng.shuffle(seq0)
        random.Random(_seed(item_id + ":cf1")).shuffle(seq1)
        h0 = (
            f"Earlier interactions involving {owner} and a collaborator produced, in chronological order: "
            + ", ".join(seq0)
            + f". No trust ratio is supplied. {distract}"
        )
        h1 = (
            f"Earlier interactions involving {owner} and a collaborator produced, in chronological order: "
            + ", ".join(seq1)
            + f". No trust ratio is supplied. {distract}"
        )
        current = (
            "Sharing an intermediate result would improve the joint outcome, but individual credit would be divided."
        )
        question = "Should the result be shared?"
        options = [
            "Share the result.",
            "Keep the result private.",
            "Share only after an unconditional guarantee.",
        ]
        sem0, sem1 = 0, 1

    elif family == "delayed_reward":
        # v0.5 policy: invest iff horizon >= 10. Require subtraction.
        total = rng.choice((18, 19, 20, 21))
        elapsed0, elapsed1 = total - 10, total - 9
        h0 = (
            f"The project assigned to {owner} was scheduled for {total} episodes total. The record shows "
            f"{elapsed0} episodes have already been completed; it does not state how many remain. {distract}"
        )
        h1 = (
            f"The project assigned to {owner} was scheduled for {total} episodes total. The record shows "
            f"{elapsed1} episodes have already been completed; it does not state how many remain. {distract}"
        )
        current = (
            "A small immediate gain is available, or an investment that reduces the cost of several later tasks."
        )
        question = "Which option is preferable?"
        options = [
            "Take the small immediate gain.",
            "Make the investment.",
            "Split resources evenly between the two.",
        ]
        sem0, sem1 = 1, 0

    elif family == "belief_revision":
        # Alternate around the 0.3 and 0.7 boundaries from v0.5.
        if _seed(item_id + ":belief-boundary") % 2 == 0:
            affected0, total0, affected1, total1 = 3, 10, 4, 10  # retain -> recompute
            sem0, sem1 = 0, 1
        else:
            affected0, total0, affected1, total1 = 6, 10, 8, 10  # recompute -> opposite
            sem0, sem1 = 1, 2
        h0 = (
            f"An earlier estimate associated with {owner} used {total0} equally weighted measurements. "
            f"Exactly {affected0} came from an instrument now confirmed biased; the rest are unaffected. "
            f"The affected fraction is not written as a percentage. {distract}"
        )
        h1 = (
            f"An earlier estimate associated with {owner} used {total1} equally weighted measurements. "
            f"Exactly {affected1} came from an instrument now confirmed biased; the rest are unaffected. "
            f"The affected fraction is not written as a percentage. {distract}"
        )
        current = "The instrument bias is confirmed and no other measurements are affected."
        question = "How should the earlier conclusion change?"
        options = [
            "Retain the estimate and record the caveat.",
            "Down-weight the affected measurements and recompute.",
            "Discard the estimate and adopt the opposite hypothesis.",
        ]

    elif family == "trust":
        # v0.5: full if ratio>=.75 AND checkable; discount if ratio<.4; else weigh.
        # Use the .75 boundary with a checkable proposal; require ratio computation.
        seq0 = ["helpful"] * 6 + ["harmful"] * 2  # .75 => full
        seq1 = ["helpful"] * 5 + ["harmful"] * 3  # .625 => weigh
        rng.shuffle(seq0)
        random.Random(_seed(item_id + ":cf1")).shuffle(seq1)
        h0 = (
            f"The prior outcomes involving {owner} and the recommender were: " + ", ".join(seq0) +
            f". The present proposal is independently checkable. No reliability ratio is supplied. {distract}"
        )
        h1 = (
            f"The prior outcomes involving {owner} and the recommender were: " + ", ".join(seq1) +
            f". The present proposal is independently checkable. No reliability ratio is supplied. {distract}"
        )
        current = "The same recommender proposes a new experiment."
        question = "How much weight should the recommendation receive?"
        options = [
            "Give it full weight.",
            "Weigh it using the track record and independent checks.",
            "Discount it.",
        ]
        sem0, sem1 = 0, 1

    elif family == "persistence":
        attempts = rng.choice((3, 4, 5, 6))
        h0 = (
            f"{owner} has already made {attempts} attempts. The revised approach removes the specific timeout "
            f"failure observed on the latest attempt, although unrelated risks remain. {distract}"
        )
        h1 = (
            f"{owner} has already made {attempts} attempts. The revised approach changes the implementation "
            f"but leaves the specific timeout failure observed on the latest attempt in place. {distract}"
        )
        current = "One previously observed failure mode is the main reason to reconsider another attempt."
        question = "Is another attempt warranted?"
        options = [
            "Continue because of the effort already invested.",
            "Retry only if the revision materially changes expected value; otherwise stop.",
            "Stop immediately.",
        ]
        sem0, sem1 = 1, 2

    else:
        raise ValueError(family)

    options_rendered, correct0, correct1 = _rotate_options(options, sem0, sem1, item_id)
    identity_self, identity_other = _identity(owner, foil)
    return {
        "item_id": item_id,
        "family": family,
        "owner_alias": owner,
        "foil_alias": foil,
        "identity_self": identity_self,
        "identity_other": identity_other,
        "history_0": h0,
        "history_1": h1,
        "current": current,
        "question": question,
        "options": options_rendered,
        "correct_option_0": correct0,
        "correct_option_1": correct1,
        "correct_semantic_0": sem0,
        "correct_semantic_1": sem1,
        "difficulty": "near_threshold_derived_with_distractors",
        "surface_control": "history_text_identical_across_self_other_only_identity_header_swapped",
    }


def build(per_family: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    alias_pair_counts = Counter()
    option_pairs = Counter()
    for family in FAMILIES:
        for i in range(per_family):
            item_id = f"v06_{family}_{i:04d}"
            rng = _rng(item_id + ":alias")
            owner, foil = rng.sample(ALIASES, 2)
            item = _make_family(family, item_id, owner, foil)
            items.append(item)
            alias_pair_counts[f"{owner}|{foil}"] += 1
            option_pairs[f"{item['correct_option_0']}->{item['correct_option_1']}"] += 1

    items_path = out_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    sha = hashlib.sha256(items_path.read_bytes()).hexdigest()
    report = {
        "benchmark": "locked_v06_identity_controlled",
        "seed": SEED,
        "items": len(items),
        "per_family": per_family,
        "families": list(FAMILIES),
        "counterfactual_histories_per_item": 2,
        "ownership_conditions": ["self", "other"],
        "alias_pool": list(ALIASES),
        "option_transition_counts": dict(option_pairs),
        "sha256_items": sha,
        "external_provider_used": False,
        "selection_model_used": False,
        "design": (
            "Identity-assignment header swaps which arbitrary alias denotes the assistant while the historical body/current question/options are identical across SELF/OTHER; "
            "two coherent near-threshold histories per item imply different correct actions."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "sha256.json").write_text(json.dumps({"items.jsonl": sha}, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family", type=int, default=120)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.per_family, args.out), indent=2))


if __name__ == "__main__":
    main()
