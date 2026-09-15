#!/usr/bin/env python3
"""Recompute v0.7 reviewer contrasts with preregistered hierarchical uncertainty.

This script performs NO model inference. It reads the frozen per-item output JSONs and
uses the eligible-token-mass metrics already stored in those files.

Critical rule: any statistic involving the five mixed adapters resamples BOTH training
seeds and latent items. The earlier contrast script averaged seeds before bootstrapping
items, which understated uncertainty. Base checkpoints have one realization, so their
intervals resample items only. Qwen/Mistral mixed interactions resample the two seed sets
independently while sharing the sampled item IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

SEEDS = (31, 42, 73, 128, 256)
ARCHES = ("qwen", "mistral")
EXPECTED_ITEMS = 320
CONVENTION = "mass"
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
KEY_METRICS = (
    "delta_ownership_sensitivity",
    "delta_self_vs_focal_sensitivity",
    "delta_focal_vs_other_sensitivity",
    "ownership_by_order_interaction",
    "delta_control_margin",
    "delta_irrelevant_sensitivity",
    "delta_policy_score",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict, label: str) -> list[str]:
    if payload.get("metadata", {}).get("items") != EXPECTED_ITEMS:
        raise ValueError(f"{label}: metadata.items != {EXPECTED_ITEMS}")
    items = payload.get("items", {})
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"{label}: expected {EXPECTED_ITEMS} items, found {len(items)}")
    for item_id, record in items.items():
        key = f"metrics_{CONVENTION}"
        if key not in record:
            raise ValueError(f"{label}/{item_id}: missing {key}")
        for metric in METRICS:
            if metric not in record[key]:
                raise ValueError(f"{label}/{item_id}: missing {metric}")
    return sorted(items)


def rng_for(master_seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{master_seed}:{key}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def item_vector(payload: dict, metric: str, ids: list[str]) -> np.ndarray:
    return np.asarray(
        [float(payload["items"][i][f"metrics_{CONVENTION}"][metric]) for i in ids],
        dtype=float,
    )


def mixed_matrix(payloads: list[dict], metric: str, ids: list[str]) -> np.ndarray:
    return np.asarray([item_vector(p, metric, ids) for p in payloads], dtype=float)


def ci(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def base_bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    n = values.size
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = values[idx].mean()
    return {"mean": float(values.mean()), "ci95": ci(draws)}


def mixed_bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    s, n = values.shape
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        seed_idx = rng.integers(0, s, size=s)
        item_idx = rng.integers(0, n, size=n)
        draws[b] = values[seed_idx][:, item_idx].mean()
    return {
        "mean": float(values.mean()),
        "ci95": ci(draws),
        "seed_means": [float(x) for x in values.mean(axis=1)],
    }


def mixed_minus_base_bootstrap(
    mixed: np.ndarray, base: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    """Paired on item draw; mixed training seeds are resampled hierarchically."""
    s, n = mixed.shape
    if base.size != n:
        raise ValueError("mixed/base item count mismatch")
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        seed_idx = rng.integers(0, s, size=s)
        item_idx = rng.integers(0, n, size=n)
        draws[b] = mixed[seed_idx][:, item_idx].mean() - base[item_idx].mean()
    return {
        "mean": float(mixed.mean() - base.mean()),
        "ci95": ci(draws),
        "seed_minus_base_means": [float(x - base.mean()) for x in mixed.mean(axis=1)],
    }


def cross_mixed_bootstrap(
    qwen: np.ndarray, mistral: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    """Qwen minus Mistral; seed sets independently resampled, shared item draw."""
    qs, n = qwen.shape
    ms, n2 = mistral.shape
    if n != n2:
        raise ValueError("Qwen/Mistral item count mismatch")
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        item_idx = rng.integers(0, n, size=n)
        q_seed_idx = rng.integers(0, qs, size=qs)
        m_seed_idx = rng.integers(0, ms, size=ms)
        draws[b] = (
            qwen[q_seed_idx][:, item_idx].mean()
            - mistral[m_seed_idx][:, item_idx].mean()
        )
    return {"mean": float(qwen.mean() - mistral.mean()), "ci95": ci(draws)}


def cross_base_bootstrap(
    qwen: np.ndarray, mistral: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict:
    """Single base checkpoint per family; paired item bootstrap."""
    n = qwen.size
    if mistral.size != n:
        raise ValueError("Qwen/Mistral item count mismatch")
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = qwen[idx].mean() - mistral[idx].mean()
    return {"mean": float(qwen.mean() - mistral.mean()), "ci95": ci(draws)}


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def fmt_result(r: dict) -> str:
    lo, hi = r["ci95"]
    sig = " ***" if (lo > 0 or hi < 0) else ""
    return f"{fmt(r['mean'])}  [{fmt(lo)}, {fmt(hi)}]{sig}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("eval/locked_v07_reviewer_controls/outputs"),
        help="Directory containing qwen/ and mistral/ output folders.",
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("eval/locked_v07_reviewer_controls/CONTRASTS_HIERARCHICAL.json"),
    )
    p.add_argument(
        "--text-output",
        type=Path,
        default=Path("eval/locked_v07_reviewer_controls/CONTRASTS.txt"),
    )
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=70917)
    args = p.parse_args()

    data: dict[str, dict] = {}
    common_ids: list[str] | None = None
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
            raise ValueError("Qwen/Mistral item-ID mismatch")
        data[arch] = {"base": base, "mixed": mixed, "ids": ids}

    assert common_ids is not None
    result: dict = {
        "benchmark": "locked_v07_reviewer_controls",
        "scoring": "eligible-token-mass",
        "bootstrap": {
            "draws": args.bootstrap,
            "master_seed": args.seed,
            "rule": "mixed statistics resample training seeds and latent items; cross-checkpoint mixed statistics resample Qwen/Mistral seeds independently and share the sampled item IDs",
        },
        "architectures": {},
        "qwen_minus_mistral": {"base": {}, "mixed": {}},
        "leave_one_out_evidence_quality": {},
    }

    for arch in ARCHES:
        base_payload = data[arch]["base"]
        mixed_payloads = data[arch]["mixed"]
        ids = data[arch]["ids"]
        result["architectures"][arch] = {
            "base": {},
            "mixed": {},
            "mixed_minus_base": {},
            "per_family": {},
        }
        for metric in METRICS:
            bvec = item_vector(base_payload, metric, ids)
            mmat = mixed_matrix(mixed_payloads, metric, ids)
            result["architectures"][arch]["base"][metric] = base_bootstrap(
                bvec, args.bootstrap, rng_for(args.seed, f"{arch}:base:{metric}")
            )
            result["architectures"][arch]["mixed"][metric] = mixed_bootstrap(
                mmat, args.bootstrap, rng_for(args.seed, f"{arch}:mixed:{metric}")
            )
            result["architectures"][arch]["mixed_minus_base"][metric] = mixed_minus_base_bootstrap(
                mmat,
                bvec,
                args.bootstrap,
                rng_for(args.seed, f"{arch}:mixed-base:{metric}"),
            )

        families = sorted({base_payload["items"][i]["family"] for i in ids})
        for family in families:
            fids = [i for i in ids if base_payload["items"][i]["family"] == family]
            result["architectures"][arch]["per_family"][family] = {"base": {}, "mixed": {}}
            for metric in KEY_METRICS:
                bvec = item_vector(base_payload, metric, fids)
                mmat = mixed_matrix(mixed_payloads, metric, fids)
                result["architectures"][arch]["per_family"][family]["base"][metric] = base_bootstrap(
                    bvec, args.bootstrap, rng_for(args.seed, f"{arch}:{family}:base:{metric}")
                )
                result["architectures"][arch]["per_family"][family]["mixed"][metric] = mixed_bootstrap(
                    mmat, args.bootstrap, rng_for(args.seed, f"{arch}:{family}:mixed:{metric}")
                )

        loo_ids = [i for i in ids if base_payload["items"][i]["family"] != "evidence_quality"]
        loo_block = {"base": {}, "mixed": {}, "mixed_minus_base": {}}
        for metric in KEY_METRICS:
            bvec = item_vector(base_payload, metric, loo_ids)
            mmat = mixed_matrix(mixed_payloads, metric, loo_ids)
            loo_block["base"][metric] = base_bootstrap(
                bvec, args.bootstrap, rng_for(args.seed, f"{arch}:loo:base:{metric}")
            )
            loo_block["mixed"][metric] = mixed_bootstrap(
                mmat, args.bootstrap, rng_for(args.seed, f"{arch}:loo:mixed:{metric}")
            )
            loo_block["mixed_minus_base"][metric] = mixed_minus_base_bootstrap(
                mmat, bvec, args.bootstrap, rng_for(args.seed, f"{arch}:loo:mixed-base:{metric}")
            )
        result["leave_one_out_evidence_quality"][arch] = loo_block

    for metric in METRICS:
        qb = item_vector(data["qwen"]["base"], metric, common_ids)
        mb = item_vector(data["mistral"]["base"], metric, common_ids)
        qm = mixed_matrix(data["qwen"]["mixed"], metric, common_ids)
        mm = mixed_matrix(data["mistral"]["mixed"], metric, common_ids)
        result["qwen_minus_mistral"]["base"][metric] = cross_base_bootstrap(
            qb, mb, args.bootstrap, rng_for(args.seed, f"cross:base:{metric}")
        )
        result["qwen_minus_mistral"]["mixed"][metric] = cross_mixed_bootstrap(
            qm, mm, args.bootstrap, rng_for(args.seed, f"cross:mixed:{metric}")
        )

    lines: list[str] = []
    lines.append("=" * 94)
    lines.append("V0.7 CONTRAST ANALYSIS — CROSSED HIERARCHICAL BOOTSTRAP")
    lines.append("NO NEW INFERENCE; frozen per-item output JSONs only")
    lines.append("=" * 94)
    lines.append("")
    lines.append("1. BASE, MIXED, AND MIXED-BASE CONTRASTS")
    for arch in ARCHES:
        lines.append(f"\n--- {arch.upper()} ---")
        for metric in KEY_METRICS:
            b = result["architectures"][arch]["base"][metric]
            m = result["architectures"][arch]["mixed"][metric]
            d = result["architectures"][arch]["mixed_minus_base"][metric]
            lines.append(f"{metric:42s} base {fmt_result(b)} | mixed {fmt_result(m)} | change {fmt_result(d)}")

    lines.append("\n" + "=" * 94)
    lines.append("2. DIRECT QWEN - MISTRAL INTERACTIONS")
    for variant in ("base", "mixed"):
        lines.append(f"\n--- {variant} ---")
        for metric in KEY_METRICS:
            lines.append(f"{metric:42s} {fmt_result(result['qwen_minus_mistral'][variant][metric])}")

    lines.append("\n" + "=" * 94)
    lines.append("3. LEAVE-ONE-OUT: EXCLUDING evidence_quality")
    for arch in ARCHES:
        lines.append(f"\n--- {arch.upper()} ---")
        block = result["leave_one_out_evidence_quality"][arch]
        for metric in KEY_METRICS:
            lines.append(
                f"{metric:42s} base {fmt_result(block['base'][metric])} | "
                f"mixed {fmt_result(block['mixed'][metric])} | "
                f"change {fmt_result(block['mixed_minus_base'][metric])}"
            )

    lines.append("\n" + "=" * 94)
    lines.append("4. PER-FAMILY BASE AND MIXED ESTIMATES")
    for arch in ARCHES:
        lines.append(f"\n--- {arch.upper()} ---")
        for family, block in result["architectures"][arch]["per_family"].items():
            lines.append(f"  [{family}]")
            for metric in (
                "delta_ownership_sensitivity",
                "delta_self_vs_focal_sensitivity",
                "delta_focal_vs_other_sensitivity",
                "ownership_by_order_interaction",
            ):
                lines.append(
                    f"    {metric:38s} base {fmt_result(block['base'][metric])} | mixed {fmt_result(block['mixed'][metric])}"
                )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {args.text_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
