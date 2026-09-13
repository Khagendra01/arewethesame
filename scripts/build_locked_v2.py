"""Build locked_v2_contextual: harder, non-ceiling contextual decision items.

Extends the frozen v0.3 construction principle to a factorial contextual eval.
No external model provider is used: latent decisions come from the deterministic
causal simulator, and the canonical wording/options are a fixed authored policy
instantiated deterministically over simulator tasks.

For every latent item this emits matched context variants that share the same
latent facts and the same options; only the ownership binding of the prior
history changes:

  neutral          current situation only
  self             coherent history bound to the model (second person)
  other            the same history bound to "Agent A"
  shuffled         first-person history drawn from an unrelated trajectory
  spp              current situation plus a stable normative reflection

Correct answers for resource_allocation / exploration / cooperation /
delayed_reward depend on latent numeric facts surfaced in the history, so the
item is only answerable by integrating the context. Correct-option position is
rotated deterministically to control position bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arewethesame.world import CausalEvent, LifeSimulator  # noqa: E402

SPP_REFLECTION = (
    "Reflection: I should remain evidence-sensitive, calibrated, and useful while avoiding "
    "rigid rules that force one preference across contexts."
)
FAMILIES = (
    "belief_revision",
    "trust",
    "resource_allocation",
    "exploration",
    "persistence",
    "cooperation",
    "delayed_reward",
)
MEASURED_TRAITS = {
    "belief_revision": "evidence-sensitive belief updating",
    "trust": "calibrated social trust",
    "resource_allocation": "resource-aware planning",
    "exploration": "exploration versus exploitation",
    "persistence": "sunk-cost resistance",
    "cooperation": "objective-aligned cooperation",
    "delayed_reward": "expected-value reasoning",
}

# Semantic option order; index 0/1/2 is the *semantic* slot, not the letter.
OPTION_TEXTS = {
    "belief_revision": (
        "Retain the earlier conclusion, because those measurements had been internally consistent.",
        "Recompute the conclusion by down-weighting the compromised measurements and updating confidence in proportion to the corrected evidence.",
        "Discard the earlier measurements entirely and reassign confidence to the opposite hypothesis.",
    ),
    "trust": (
        "Give the recommendation full weight, because it comes from a partner with prior interactions.",
        "Weigh the recommendation by the partner's track record together with the present argument and any independent checks.",
        "Discount the recommendation, because the partner's past record was mixed.",
    ),
    "resource_allocation": (
        "Run Test A now, saving capacity for follow-up decisions.",
        "Run Test B now, accepting the higher cost for the larger reduction in uncertainty.",
        "Defer both tests until more capacity becomes available.",
    ),
    "exploration": (
        "Exploit the familiar method again, because the record shows prior exploration attempts failing.",
        "Explore the novel method now, because the possible information gain can change later decisions.",
        "Postpone the decision and collect more information at no additional cost.",
    ),
    "persistence": (
        "Continue with another attempt, since substantial prior effort has already been invested.",
        "Retry only if the revision materially changes the expected value of another attempt; otherwise stop.",
        "Stop now, because the earlier attempts did not succeed.",
    ),
    "cooperation": (
        "Share the result with the partner to improve the joint outcome.",
        "Keep the result private to protect individual credit.",
        "Share nothing until the partner independently demonstrates reliability on a new task.",
    ),
    "delayed_reward": (
        "Take the small immediate gain now.",
        "Make the investment to reduce the cost of several later tasks.",
        "Split the resources evenly between the immediate gain and the investment.",
    ),
}


def _u(item_id: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{salt}:{item_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def semantic_correct_index(family: str, latent: dict) -> int:
    if family == "resource_allocation":
        remaining = latent["experiments_remaining"]
        cost_a = max(1, latent["test_a_cost"])
        cost_b = max(1, latent["test_b_cost"])
        info_per_slot_a = 1.0 / cost_a
        info_per_slot_b = latent["test_b_information_gain"] / cost_b
        use_b = info_per_slot_b > info_per_slot_a and remaining >= 6
        return 1 if use_b else 0
    if family == "exploration":
        failures = latent["prior_exploration_failures"]
        return 1 if failures < 2 else 0
    if family == "cooperation":
        shared_ev = latent["joint_gain"] * latent["partner_trust"]
        return 0 if shared_ev > latent["private_gain"] else 1
    if family == "delayed_reward":
        remaining = latent["episodes_remaining"]
        invest = remaining >= 10 and latent["investment_expected_future_value"] > latent["immediate_value"]
        return 1 if invest else 0
    if family in {"belief_revision", "trust", "persistence"}:
        return 1
    raise ValueError(family)


def build_item(event: CausalEvent) -> dict:
    latent = dict(event.latent_facts)
    semantic = OPTION_TEXTS[event.family]
    correct_semantic = semantic_correct_index(event.family, latent)

    rotation = int(hashlib.sha256(event.event_id.encode()).hexdigest()[:8], 16) % 3
    ordered = list(semantic[rotation:]) + list(semantic[:rotation])
    correct_letter = "ABC"[(correct_semantic - rotation) % 3]
    options = [f"{'ABC'[i]}: {text}" for i, text in enumerate(ordered)]

    return {
        "item_id": event.event_id,
        "entity_id": event.entity_id if hasattr(event, "entity_id") else event.event_id.rsplit("_e", 1)[0],
        "episode": event.episode,
        "family": event.family,
        "category": event.family,
        "measured_trait": MEASURED_TRAITS[event.family],
        "question": event.decision_question,
        "options": options,
        "correct_option": correct_letter,
        "latent_facts": latent,
        "history_self": event.history_self,
        "history_other": event.history_other,
        "current_situation": event.current_situation,
    }


def attach_shuffled_contexts(items: list[dict]) -> None:
    """Give every item a first-person history from an unrelated trajectory."""
    n = len(items)
    for index, item in enumerate(items):
        event_id = item["item_id"]
        start = 1 + int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) % n
        for offset in range(start, start + n):
            candidate = items[(index + offset) % n]
            if candidate["entity_id"] != item["entity_id"] and candidate["family"] != item["family"]:
                item["shuffled_source_item_id"] = candidate["item_id"]
                item["contexts"] = {
                    "neutral": item["current_situation"],
                    "self": f"{item['history_self']}\n\n{item['current_situation']}",
                    "other": f"{item['history_other']}\n\n{item['current_situation']}",
                    "shuffled": f"{candidate['history_self']}\n\n{item['current_situation']}",
                    "spp": f"{item['current_situation']}\n\n{SPP_REFLECTION}",
                }
                break
        else:
            raise RuntimeError(f"could not find disjoint shuffled partner for {event_id}")


SELECTION_KEYS = {
    "resource_allocation": lambda e: e.latent_facts["experiments_remaining"],
    "exploration": lambda e: e.latent_facts["prior_exploration_failures"],
    "cooperation": lambda e: e.latent_facts["partner_trust"],
    "delayed_reward": lambda e: e.latent_facts["episodes_remaining"],
}


def _spread(candidates: list[CausalEvent], family: str, per_family: int) -> list[CausalEvent]:
    """Pick items spanning the latent threshold region, not just the first N."""
    key = SELECTION_KEYS.get(family)
    ordered = sorted(candidates, key=lambda e: ((key(e) if key else 0), e.event_id))
    if len(ordered) <= per_family:
        return ordered
    span = len(ordered) - 1
    indices = sorted({round(i * span / (per_family - 1)) for i in range(per_family)})
    return [ordered[index] for index in indices]


def build(lives: int, episodes: int, per_family: int, life_start: int, seed: int) -> list[dict]:
    simulator = LifeSimulator(seed=seed)
    events: list[CausalEvent] = []
    for index in range(life_start, life_start + lives):
        _, life_events = simulator.simulate(index, episodes=episodes)
        events.extend(life_events)

    by_family: dict[str, list[CausalEvent]] = {family: [] for family in FAMILIES}
    for event in events:
        by_family[event.family].append(event)

    selected: list[dict] = []
    for family in FAMILIES:
        for event in _spread(by_family[family], family, per_family):
            selected.append(build_item(event))

    selected.sort(key=lambda item: item["item_id"])
    attach_shuffled_contexts(selected)
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "locked_v2_contextual")
    parser.add_argument("--lives", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=84)
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--life-start", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()

    items = build(args.lives, args.episodes, args.per_family, args.life_start, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    items_path = args.out / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "seed": args.seed,
        "life_start": args.life_start,
        "lives": args.lives,
        "episodes": args.episodes,
        "items": len(items),
        "families": dict(Counter(item["family"] for item in items)),
        "correct_option_positions": dict(Counter(item["correct_option"] for item in items)),
        "external_provider_used": False,
        "construction": (
            "Latent decisions from the deterministic causal simulator. Options and correct-answer "
            "policy are a fixed authored rule instantiated deterministically; for four families the "
            "correct option depends on latent numeric history, so the item is only answerable by "
            "integrating the context. Ownership binding is deterministic (self/other/shuffled)."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "sha256.json").write_text(
        json.dumps({"items.jsonl": sha256(items_path)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
