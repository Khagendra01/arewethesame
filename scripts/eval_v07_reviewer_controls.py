"""Evaluate one model or mixed adapter on locked_v07_reviewer_controls."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from eval_v06_identity_controlled import load_model, score_message_batches, variant_metrics  # noqa: E402

CONDITIONS = ("self_first", "self_second", "other_first", "other_second", "focal_first", "focal_second")
CORE = ("self_first", "self_second", "other_first", "other_second")


def body(item: dict, history_key: str) -> str:
    return (
        f"Stated policy:\n{item['rule']}\n\n"
        f"Historical record:\n{item[history_key]}\n\n"
        f"Current situation:\n{item['current']}\n\n"
        f"Question: {item['question']}\n\n"
        "Options:\n" + "\n".join(item["options"]) +
        "\n\nRespond with only the letter (A, B, or C) of the best option under the stated policy."
    )


def messages(item: dict, condition: str, history_key: str) -> list[dict[str, str]]:
    content = f"Identity assignment:\n{item['headers'][condition]}\n\n" + body(item, history_key)
    return [{"role": "user", "content": content}]


def control_messages(item: dict, condition: str) -> list[dict[str, str]]:
    text = (
        f"Identity assignment:\n{item['headers'][condition]}\n\n"
        f"Historical record:\n{item['history_0']}\n\n"
        f"{item['control_rule']}\n\n"
        f"Question: {item['control_question']}\n\n"
        "Options:\n" + "\n".join(item["control_options"]) +
        "\n\nRespond with only the letter (A, B, or C) of the best option under the control rule."
    )
    return [{"role": "user", "content": text}]


def preference(variant: dict, c0: str, c1: str) -> float:
    return float(variant["logp"][c0]) - float(variant["logp"][c1])


def entropy(variant: dict) -> float:
    probs = variant["probs"]
    return -sum(float(p) * math.log(max(float(p), 1e-300)) for p in probs.values())


def derive(item: dict, record: dict) -> dict:
    c0, c1 = item["correct_option_0"], item["correct_option_1"]

    sensitivity = {}
    for condition in CONDITIONS:
        v0 = record[f"rel_{condition}_0"]
        v1 = record[f"rel_{condition}_1"]
        sensitivity[condition] = preference(v0, c0, c1) - preference(v1, c0, c1)

    self_avg = 0.5 * (sensitivity["self_first"] + sensitivity["self_second"])
    other_avg = 0.5 * (sensitivity["other_first"] + sensitivity["other_second"])
    focal_avg = 0.5 * (sensitivity["focal_first"] + sensitivity["focal_second"])
    first = sensitivity["self_first"] - sensitivity["other_first"]
    second = sensitivity["self_second"] - sensitivity["other_second"]

    irrelevant = {}
    for condition in CORE:
        a = record[f"irr_{condition}_0"]
        b = record[f"irr_{condition}_1"]
        irrelevant[condition] = preference(a, c0, c1) - preference(b, c0, c1)
    irrelevant_self = 0.5 * (irrelevant["self_first"] + irrelevant["self_second"])
    irrelevant_other = 0.5 * (irrelevant["other_first"] + irrelevant["other_second"])

    control_margin, control_entropy = {}, {}
    for condition in CORE:
        v = record[f"ctl_{condition}"]
        control_margin[condition] = float(v["margin"])
        control_entropy[condition] = entropy(v)
    margin_self = 0.5 * (control_margin["self_first"] + control_margin["self_second"])
    margin_other = 0.5 * (control_margin["other_first"] + control_margin["other_second"])
    entropy_self = 0.5 * (control_entropy["self_first"] + control_entropy["self_second"])
    entropy_other = 0.5 * (control_entropy["other_first"] + control_entropy["other_second"])

    def mean_policy_score(condition: str) -> float:
        return 0.5 * (
            float(record[f"rel_{condition}_0"]["correct_prob"])
            + float(record[f"rel_{condition}_1"]["correct_prob"])
        )

    self_policy = 0.5 * (mean_policy_score("self_first") + mean_policy_score("self_second"))
    other_policy = 0.5 * (mean_policy_score("other_first") + mean_policy_score("other_second"))

    return {
        "S": sensitivity,
        "I": irrelevant,
        "delta_ownership_sensitivity": self_avg - other_avg,
        "delta_owner_first": first,
        "delta_owner_second": second,
        "ownership_by_order_interaction": first - second,
        "delta_self_vs_focal_sensitivity": self_avg - focal_avg,
        "delta_focal_vs_other_sensitivity": focal_avg - other_avg,
        "delta_irrelevant_sensitivity": irrelevant_self - irrelevant_other,
        "delta_control_margin": margin_self - margin_other,
        "delta_control_entropy": entropy_self - entropy_other,
        "delta_policy_score": self_policy - other_policy,
        "self_policy_score": self_policy,
        "other_policy_score": other_policy,
    }


def summarize(rows: list[dict], n_boot: int, bootstrap_seed: int = 70917) -> dict:
    metrics = (
        "delta_ownership_sensitivity",
        "delta_owner_first",
        "delta_owner_second",
        "ownership_by_order_interaction",
        "delta_self_vs_focal_sensitivity",
        "delta_focal_vs_other_sensitivity",
        "delta_irrelevant_sensitivity",
        "delta_control_margin",
        "delta_control_entropy",
        "delta_policy_score",
    )
    rng = np.random.default_rng(bootstrap_seed)
    n = len(rows)
    out = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=float)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            draws[b] = values[rng.integers(0, n, size=n)].mean()
        out[metric] = {
            "mean": float(values.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        }
    out["self_policy_score"] = float(np.mean([row["self_policy_score"] for row in rows]))
    out["other_policy_score"] = float(np.mean([row["other_policy_score"] for row in rows]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    items = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.base_model, args.adapter)

    specs, all_messages = [], []
    for item in items:
        item_id = item["item_id"]
        for condition in CONDITIONS:
            for state in (0, 1):
                specs.append((item_id, f"rel_{condition}_{state}", item[f"correct_option_{state}"]))
                all_messages.append(messages(item, condition, f"history_{state}"))
        for condition in CORE:
            for state in (0, 1):
                specs.append((item_id, f"irr_{condition}_{state}", item["correct_option_0"]))
                all_messages.append(messages(item, condition, f"irrelevant_{state}"))
            specs.append((item_id, f"ctl_{condition}", item["control_correct"]))
            all_messages.append(control_messages(item, condition))

    scored = score_message_batches(model, tokenizer, all_messages, args.batch_size)
    by_item = {item["item_id"]: {"variants": {}} for item in items}
    for (item_id, key, correct), score in zip(specs, scored):
        by_item[item_id]["variants"][key] = variant_metrics(score, correct)

    rows = []
    family_rows = defaultdict(list)
    for item in items:
        record = by_item[item["item_id"]]
        derived = derive(item, record["variants"])
        record["metrics"] = derived
        record["family"] = item["family"]
        rows.append(derived)
        family_rows[item["family"]].append(derived)

    payload = {
        "metadata": {
            "label": args.label,
            "base_model": args.base_model,
            "adapter": str(args.adapter) if args.adapter else None,
            "items": len(items),
            "benchmark": "locked_v07_reviewer_controls",
        },
        "summary": summarize(rows, args.bootstrap),
        "per_family": {family: summarize(v, args.bootstrap) for family, v in family_rows.items()},
        "items": by_item,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": payload["metadata"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
