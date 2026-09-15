"""Aggregate v0.7 reviewer-control results across independent training seeds.

Primary reporting uses eligible-token-mass scoring. Legacy max-over-token-realizations
is aggregated as a scoring robustness analysis from the same inference outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (31, 42, 73, 128, 256)
ARCHES = ("qwen", "mistral")
EXPECTED_ITEMS = 320
METRICS = (
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
CONVENTIONS = ("mass", "max")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict, label: str) -> list[str]:
    if payload["metadata"]["items"] != EXPECTED_ITEMS or len(payload["items"]) != EXPECTED_ITEMS:
        raise ValueError(f"{label}: incomplete")
    for item_id, record in payload["items"].items():
        if "metrics_mass" not in record or "metrics_max" not in record:
            raise ValueError(f"{label}/{item_id}: missing scoring convention")
    return sorted(payload["items"])


def matrix(payloads: list[dict], metric: str, ids: list[str], convention: str) -> np.ndarray:
    return np.array(
        [
            [float(payload["items"][item_id][f"metrics_{convention}"][metric]) for item_id in ids]
            for payload in payloads
        ],
        dtype=float,
    )


def hierarchical_bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    seeds, n = values.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = values[
            rng.integers(0, seeds, size=seeds)
        ][:, rng.integers(0, n, size=n)].mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "seed_means": [float(x) for x in values.mean(axis=1)],
    }


def base_bootstrap(
    payload: dict,
    metric: str,
    ids: list[str],
    convention: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    values = np.array(
        [payload["items"][item_id][f"metrics_{convention}"][metric] for item_id in ids],
        dtype=float,
    )
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = values[rng.integers(0, len(values), size=len(values))].mean()
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def architecture_interaction(
    qwen: list[dict],
    mistral: list[dict],
    metric: str,
    ids: list[str],
    convention: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    q = matrix(qwen, metric, ids, convention)
    m = matrix(mistral, metric, ids, convention)
    q_seeds, n = q.shape
    m_seeds, _ = m.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        item_idx = rng.integers(0, n, size=n)
        draws[b] = (
            q[rng.integers(0, q_seeds, size=q_seeds)][:, item_idx].mean()
            - m[rng.integers(0, m_seeds, size=m_seeds)][:, item_idx].mean()
        )
    return {
        "mean": float(q.mean() - m.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "note": "numeric seed labels are not paired; seeds are resampled independently and items are shared",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rng = np.random.default_rng(70917)

    data = {}
    common_ids = None
    for arch in ARCHES:
        base = load(args.root / arch / "base.json")
        mixed = [load(args.root / arch / f"mixed_s{s}.json") for s in SEEDS]
        ids = validate(base, f"{arch}/base")
        for seed, payload in zip(SEEDS, mixed):
            if validate(payload, f"{arch}/mixed_s{seed}") != ids:
                raise ValueError(f"{arch}/mixed_s{seed}: item-ID mismatch")
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError("architecture item-ID mismatch")
        data[arch] = (base, mixed, ids)

    result = {
        "benchmark": "locked_v07_reviewer_controls",
        "completeness": "PASS",
        "conventions": {},
    }
    for convention in CONVENTIONS:
        block = {"architectures": {}, "qwen_minus_mistral": {}}
        for arch, (base, mixed, ids) in data.items():
            block["architectures"][arch] = {"base": {}, "mixed": {}, "per_family": {}}
            for metric in METRICS:
                block["architectures"][arch]["base"][metric] = base_bootstrap(
                    base, metric, ids, convention, args.bootstrap, rng
                )
                block["architectures"][arch]["mixed"][metric] = hierarchical_bootstrap(
                    matrix(mixed, metric, ids, convention), args.bootstrap, rng
                )
            families = sorted(set(mixed[0]["items"][i]["family"] for i in ids))
            for family in families:
                family_ids = [i for i in ids if mixed[0]["items"][i]["family"] == family]
                block["architectures"][arch]["per_family"][family] = {
                    metric: hierarchical_bootstrap(
                        matrix(mixed, metric, family_ids, convention), args.bootstrap, rng
                    )
                    for metric in METRICS
                }
        for metric in METRICS:
            block["qwen_minus_mistral"][metric] = architecture_interaction(
                data["qwen"][1],
                data["mistral"][1],
                metric,
                common_ids,
                convention,
                args.bootstrap,
                rng,
            )
        result["conventions"][convention] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    primary = result["conventions"]["mass"]
    print(
        json.dumps(
            {
                "primary_scoring": "eligible-token-mass",
                "qwen_primary": primary["architectures"]["qwen"]["mixed"]["delta_ownership_sensitivity"],
                "mistral_primary": primary["architectures"]["mistral"]["mixed"]["delta_ownership_sensitivity"],
                "qwen_vs_mistral": primary["qwen_minus_mistral"]["delta_ownership_sensitivity"],
                "qwen_order_interaction": primary["architectures"]["qwen"]["mixed"]["ownership_by_order_interaction"],
                "qwen_confidence_control": primary["architectures"]["qwen"]["mixed"]["delta_control_margin"],
                "qwen_irrelevant_control": primary["architectures"]["qwen"]["mixed"]["delta_irrelevant_sensitivity"],
                "qwen_self_vs_focal": primary["architectures"]["qwen"]["mixed"]["delta_self_vs_focal_sensitivity"],
                "legacy_max_primary_qwen": result["conventions"]["max"]["architectures"]["qwen"]["mixed"]["delta_ownership_sensitivity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
