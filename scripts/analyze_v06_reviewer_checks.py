"""Reviewer-targeted reanalysis for v0.6 result JSONs.

Consumes the existing result directory:
  ROOT/qwen/base.json
  ROOT/qwen/mixed_s{31,42,73,128,256}.json
  ROOT/mistral/base.json
  ROOT/mistral/mixed_s{31,42,73,128,256}.json

Adds analyses requested by reviewer critique without changing the frozen v0.6 estimand:
- strict completeness checks (840 common IDs, four variants per item)
- base -> MIXED paired contrast for delta_sensitivity and secondary metrics
- direct Qwen - Mistral interaction (independent seed resampling, shared item resampling)
- per-family hierarchical estimates
- leave-one-seed-out robustness
- ownership-induced decision-switch rates and absolute score changes
- per-variant probability/entropy/saturation diagnostics

No result is silently subset to an intersection: missing items are fatal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (31, 42, 73, 128, 256)
ARCHES = ("qwen", "mistral")
METRICS = ("delta_prob", "delta_margin", "delta_accuracy", "delta_sensitivity")
EXPECTED_ITEMS = 840
EXPECTED_VARIANTS = {"self_0", "self_1", "other_0", "other_1"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_payload(payload: dict, label: str) -> list[str]:
    if payload.get("metadata", {}).get("items") != EXPECTED_ITEMS:
        raise ValueError(f"{label}: metadata.items != {EXPECTED_ITEMS}")
    items = payload.get("items", {})
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"{label}: found {len(items)} item records, expected {EXPECTED_ITEMS}")
    ids = sorted(items)
    for item_id, rec in items.items():
        variants = set(rec.get("variants", {}))
        if variants != EXPECTED_VARIANTS:
            raise ValueError(f"{label}/{item_id}: variants={sorted(variants)}")
        for vname in EXPECTED_VARIANTS:
            v = rec["variants"][vname]
            probs = v.get("probs") or v.get("prob")
            if probs is None or set(probs) != {"A", "B", "C"}:
                raise ValueError(f"{label}/{item_id}/{vname}: invalid probability record")
            if not math.isclose(sum(float(x) for x in probs.values()), 1.0, abs_tol=1e-5):
                raise ValueError(f"{label}/{item_id}/{vname}: probabilities do not sum to 1")
    return ids


def get_probs(v: dict) -> dict[str, float]:
    return {k: float(x) for k, x in (v.get("probs") or v.get("prob")).items()}


def get_metric_matrix(payloads: list[dict], metric: str, ids: list[str]) -> np.ndarray:
    return np.array(
        [[float(p["items"][i]["metrics"][metric]) for i in ids] for p in payloads],
        dtype=np.float64,
    )


def hierarchical_stat(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    s, n = matrix.shape
    seed_means = matrix.mean(axis=1)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seed_idx = rng.integers(0, s, size=s)
        item_idx = rng.integers(0, n, size=n)
        draws[b] = matrix[seed_idx][:, item_idx].mean()
    return {
        "mean": float(matrix.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "seed_means": [float(x) for x in seed_means],
        "seed_sd": float(seed_means.std(ddof=1)) if s > 1 else 0.0,
        "n_seeds": int(s),
        "n_items": int(n),
    }


def base_stat(payload: dict, metric: str, ids: list[str], n_boot: int, rng: np.random.Generator) -> dict:
    values = np.array([float(payload["items"][i]["metrics"][metric]) for i in ids])
    draws = np.empty(n_boot)
    for b in range(n_boot):
        item_idx = rng.integers(0, len(values), size=len(values))
        draws[b] = values[item_idx].mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def post_minus_base(
    mixed: list[dict],
    base: dict,
    metric: str,
    ids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    mixed_matrix = get_metric_matrix(mixed, metric, ids)
    base_values = np.array([float(base["items"][i]["metrics"][metric]) for i in ids], dtype=np.float64)
    point = float(mixed_matrix.mean() - base_values.mean())
    s, n = mixed_matrix.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        seed_idx = rng.integers(0, s, size=s)
        item_idx = rng.integers(0, n, size=n)
        draws[b] = mixed_matrix[seed_idx][:, item_idx].mean() - base_values[item_idx].mean()
    return {
        "mean": point,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def architecture_interaction(
    qwen: list[dict],
    mistral: list[dict],
    metric: str,
    ids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    q = get_metric_matrix(qwen, metric, ids)
    m = get_metric_matrix(mistral, metric, ids)
    point = float(q.mean() - m.mean())
    qs, n = q.shape
    ms, n2 = m.shape
    if n != n2:
        raise ValueError("architecture matrices have different item counts")
    draws = np.empty(n_boot)
    for b in range(n_boot):
        q_seed_idx = rng.integers(0, qs, size=qs)
        m_seed_idx = rng.integers(0, ms, size=ms)
        item_idx = rng.integers(0, n, size=n)
        draws[b] = q[q_seed_idx][:, item_idx].mean() - m[m_seed_idx][:, item_idx].mean()
    return {
        "mean": point,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "note": "Qwen and Mistral seeds are resampled independently; the benchmark item resample is shared.",
    }


def per_family(
    payloads: list[dict],
    metric: str,
    ids: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    id_to_family = {i: payloads[0]["items"][i]["family"] for i in ids}
    result = {}
    for family in sorted(set(id_to_family.values())):
        family_ids = [i for i in ids if id_to_family[i] == family]
        result[family] = hierarchical_stat(get_metric_matrix(payloads, metric, family_ids), n_boot, rng)
    return result


def leave_one_seed_out(payloads: list[dict], metric: str, ids: list[str]) -> list[dict]:
    result = []
    for j, seed in enumerate(SEEDS):
        kept = payloads[:j] + payloads[j + 1 :]
        matrix = get_metric_matrix(kept, metric, ids)
        result.append({"left_out_seed": seed, "mean": float(matrix.mean())})
    return result


def entropy(probs: dict[str, float]) -> float:
    return -sum(p * math.log(max(p, 1e-300)) for p in probs.values())


def diagnostics(payloads: list[dict], ids: list[str]) -> dict:
    per_seed = []
    for seed, payload in zip(SEEDS, payloads):
        switches = []
        absolute_correct_prob_changes = []
        self_probs, other_probs = [], []
        self_entropy, other_entropy = [], []
        self_abs_margin, other_abs_margin = [], []
        for item_id in ids:
            rec = payload["items"][item_id]
            c0, c1 = rec["correct_option_0"], rec["correct_option_1"]
            for state, correct in ((0, c0), (1, c1)):
                self_variant = rec["variants"][f"self_{state}"]
                other_variant = rec["variants"][f"other_{state}"]
                sp, op = get_probs(self_variant), get_probs(other_variant)
                switches.append(float(self_variant["choice"] != other_variant["choice"]))
                absolute_correct_prob_changes.append(abs(float(sp[correct]) - float(op[correct])))
                self_probs.append(float(sp[correct]))
                other_probs.append(float(op[correct]))
                self_entropy.append(entropy(sp))
                other_entropy.append(entropy(op))
                self_abs_margin.append(abs(float(self_variant["margin"])))
                other_abs_margin.append(abs(float(other_variant["margin"])))

        def distribution(values: list[float]) -> dict:
            arr = np.array(values, dtype=float)
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "frac_le_005": float((arr <= 0.05).mean()),
                "frac_ge_095": float((arr >= 0.95).mean()),
                "frac_ge_099": float((arr >= 0.99).mean()),
            }

        per_seed.append(
            {
                "seed": seed,
                "ownership_argmax_switch_rate": float(np.mean(switches)),
                "mean_abs_correct_prob_change": float(np.mean(absolute_correct_prob_changes)),
                "self_correct_prob_distribution": distribution(self_probs),
                "other_correct_prob_distribution": distribution(other_probs),
                "self_mean_entropy": float(np.mean(self_entropy)),
                "other_mean_entropy": float(np.mean(other_entropy)),
                "self_mean_abs_margin": float(np.mean(self_abs_margin)),
                "other_mean_abs_margin": float(np.mean(other_abs_margin)),
            }
        )

    scalar_keys = (
        "ownership_argmax_switch_rate",
        "mean_abs_correct_prob_change",
        "self_mean_entropy",
        "other_mean_entropy",
        "self_mean_abs_margin",
        "other_mean_abs_margin",
    )
    pooled = {k: float(np.mean([row[k] for row in per_seed])) for k in scalar_keys}
    return {"per_seed": per_seed, "mean_across_seeds": pooled}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rng = np.random.default_rng(60915)

    data = {}
    reference_ids = None
    for arch in ARCHES:
        base = load(args.root / arch / "base.json")
        mixed = [load(args.root / arch / f"mixed_s{s}.json") for s in SEEDS]
        ids = validate_payload(base, f"{arch}/base")
        for seed, payload in zip(SEEDS, mixed):
            payload_ids = validate_payload(payload, f"{arch}/mixed_s{seed}")
            if payload_ids != ids:
                raise ValueError(f"{arch}/mixed_s{seed}: item IDs differ from base")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Qwen and Mistral item IDs differ")
        data[arch] = (base, mixed, ids)

    result = {
        "completeness": {"items": EXPECTED_ITEMS, "variants_per_item": 4, "status": "PASS"},
        "architectures": {},
    }
    for arch, (base, mixed, ids) in data.items():
        block = {
            "base": {},
            "mixed": {},
            "post_minus_base": {},
            "per_family": {},
            "leave_one_seed_out": {},
            "diagnostics": diagnostics(mixed, ids),
        }
        for metric in METRICS:
            block["base"][metric] = base_stat(base, metric, ids, args.bootstrap, rng)
            block["mixed"][metric] = hierarchical_stat(
                get_metric_matrix(mixed, metric, ids), args.bootstrap, rng
            )
            block["post_minus_base"][metric] = post_minus_base(
                mixed, base, metric, ids, args.bootstrap, rng
            )
            block["per_family"][metric] = per_family(mixed, metric, ids, args.bootstrap, rng)
            block["leave_one_seed_out"][metric] = leave_one_seed_out(mixed, metric, ids)
        result["architectures"][arch] = block

    result["qwen_minus_mistral"] = {
        metric: architecture_interaction(
            data["qwen"][1], data["mistral"][1], metric, reference_ids, args.bootstrap, rng
        )
        for metric in METRICS
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "completeness": result["completeness"],
                "qwen_base_sensitivity": result["architectures"]["qwen"]["base"]["delta_sensitivity"],
                "qwen_mixed_sensitivity": result["architectures"]["qwen"]["mixed"]["delta_sensitivity"],
                "qwen_post_minus_base": result["architectures"]["qwen"]["post_minus_base"]["delta_sensitivity"],
                "mistral_base_sensitivity": result["architectures"]["mistral"]["base"]["delta_sensitivity"],
                "mistral_mixed_sensitivity": result["architectures"]["mistral"]["mixed"]["delta_sensitivity"],
                "mistral_post_minus_base": result["architectures"]["mistral"]["post_minus_base"]["delta_sensitivity"],
                "qwen_minus_mistral": result["qwen_minus_mistral"]["delta_sensitivity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
