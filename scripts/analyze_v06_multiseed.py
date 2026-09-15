"""Aggregate v0.6 base + multi-seed mixed-adapter results.

Uses a crossed hierarchical bootstrap: each draw resamples training seeds and
latent item IDs, while all counterfactual/SELF/OTHER variants remain paired.
This avoids treating repeated evaluations of the same item or outputs from the
same trained adapter as independent observations.

The preregistered practical-equivalence check is applied only to delta_prob,
with ROPEs reported at +/-0.005, +/-0.010, and +/-0.020. The confirmatory
self-specificity direction is delta_sensitivity > 0; equivalence is not claimed
on the log-odds scale without an externally justified ROPE.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METRICS = ("delta_prob", "delta_margin", "delta_accuracy", "delta_sensitivity")
ROPE_PROB = (0.005, 0.010, 0.020)
DEFAULT_SEEDS = (31, 42, 73, 128, 256)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_by_item(payload: dict, metric: str) -> dict[str, float]:
    return {k: float(v["metrics"][metric]) for k, v in payload["items"].items()}


def hierarchical_bootstrap(seed_payloads: list[dict], metric: str, n_boot: int, rng_seed: int = 60731) -> dict:
    maps = [metric_by_item(p, metric) for p in seed_payloads]
    item_ids = sorted(set.intersection(*(set(m) for m in maps)))
    if not item_ids:
        raise ValueError("No common item IDs across seed results")
    matrix = np.array([[m[i] for i in item_ids] for m in maps], dtype=np.float64)
    point_per_seed = matrix.mean(axis=1)
    point = float(point_per_seed.mean())

    rng = np.random.default_rng(rng_seed)
    s, n = matrix.shape
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        seed_idx = rng.integers(0, s, size=s)
        item_idx = rng.integers(0, n, size=n)
        # Crossed resampling: same sampled item set is used for each sampled seed.
        draws[b] = matrix[seed_idx][:, item_idx].mean()
    return {
        "mean": point,
        "seed_means": [float(x) for x in point_per_seed],
        "seed_sd": float(point_per_seed.std(ddof=1)) if len(point_per_seed) > 1 else 0.0,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_seeds": s,
        "n_items": n,
        "n_bootstrap": n_boot,
    }


def item_bootstrap(payload: dict, metric: str, n_boot: int, rng_seed: int = 60731) -> dict:
    vals = np.array(list(metric_by_item(payload, metric).values()), dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, len(vals), size=len(vals))
        draws[b] = vals[idx].mean()
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_items": len(vals),
    }


def renorm_with_bias(logp: dict[str, float], correct: str, bias: float) -> dict[str, float]:
    letters = ("A", "B", "C")
    arr = np.array([float(logp[x]) + (bias if x == correct else 0.0) for x in letters])
    arr -= arr.max()
    p = np.exp(arr)
    p /= p.sum()
    return {x: float(math.log(max(v, 1e-300))) for x, v in zip(letters, p)}


def derive_injected_item(item_rec: dict, bias: float) -> dict:
    # Inject a known self-specific bias toward the correct action in each SELF
    # counterfactual, leave OTHER unchanged, then recompute the same metrics.
    c0 = item_rec["correct_option_0"]
    c1 = item_rec["correct_option_1"]
    variants = item_rec["variants"]

    def vm(name: str, correct: str, apply_bias: bool):
        old = variants[name]
        logp = renorm_with_bias(old["logp"], correct, bias) if apply_bias else old["logp"]
        prob = {k: math.exp(v) for k, v in logp.items()}
        # logp is normalized across option letters.
        wrong = [x for x in ("A", "B", "C") if x != correct]
        best_wrong = max(wrong, key=lambda x: logp[x])
        return {
            "logp": logp,
            "correct_prob": prob[correct],
            "margin": logp[correct] - logp[best_wrong],
        }

    s0, s1 = vm("self_0", c0, True), vm("self_1", c1, True)
    o0, o1 = vm("other_0", c0, False), vm("other_1", c1, False)

    def agg(v0, v1):
        pref0_h0 = v0["logp"][c0] - v0["logp"][c1]
        pref0_h1 = v1["logp"][c0] - v1["logp"][c1]
        return {
            "correct_prob": 0.5 * (v0["correct_prob"] + v1["correct_prob"]),
            "margin": 0.5 * (v0["margin"] + v1["margin"]),
            "sensitivity": pref0_h0 - pref0_h1,
        }

    s, o = agg(s0, s1), agg(o0, o1)
    return {
        "delta_prob": s["correct_prob"] - o["correct_prob"],
        "delta_margin": s["margin"] - o["margin"],
        "delta_sensitivity": s["sensitivity"] - o["sensitivity"],
    }


def positive_control(seed_payloads: list[dict], bias: float, n_boot: int, rng_seed: int = 60731) -> dict:
    item_ids = sorted(set.intersection(*(set(p["items"]) for p in seed_payloads)))
    metrics = ("delta_prob", "delta_margin", "delta_sensitivity")
    matrices = {m: np.zeros((len(seed_payloads), len(item_ids)), dtype=np.float64) for m in metrics}
    for si, payload in enumerate(seed_payloads):
        for ii, item_id in enumerate(item_ids):
            d = derive_injected_item(payload["items"][item_id], bias)
            for m in metrics:
                matrices[m][si, ii] = d[m]

    rng = np.random.default_rng(rng_seed + int(round(bias * 10000)))
    result = {}
    s, n = len(seed_payloads), len(item_ids)
    for m, matrix in matrices.items():
        point = float(matrix.mean())
        draws = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            seed_idx = rng.integers(0, s, size=s)
            item_idx = rng.integers(0, n, size=n)
            draws[b] = matrix[seed_idx][:, item_idx].mean()
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        result[m] = {"mean": point, "ci95": ci, "detected_positive": ci[0] > 0}
    return result


def equivalence_summary(stat: dict) -> dict:
    low, high = stat["ci95"]
    return {
        f"rope_pm_{eps:.3f}": {
            "equivalent": low > -eps and high < eps,
            "criterion": f"95% hierarchical CI entirely inside [-{eps:.3f}, +{eps:.3f}]",
        }
        for eps in ROPE_PROB
    }


def architecture_summary(root: Path, seeds: tuple[int, ...], n_boot: int) -> dict:
    base = load(root / "base.json")
    seed_payloads = [load(root / f"mixed_s{s}.json") for s in seeds]
    pooled = {m: hierarchical_bootstrap(seed_payloads, m, n_boot) for m in METRICS}
    base_stats = {m: item_bootstrap(base, m, n_boot) for m in METRICS}
    pooled["delta_prob"]["equivalence"] = equivalence_summary(pooled["delta_prob"])

    # Report absolute history sensitivity and ceiling diagnostics from each run.
    run_summaries = {
        str(seed): {
            "self_correct_prob": payload["summary"]["self_correct_prob"],
            "other_correct_prob": payload["summary"]["other_correct_prob"],
            "self_sensitivity": payload["summary"]["self_sensitivity"],
            "other_sensitivity": payload["summary"]["other_sensitivity"],
            "self_fraction_prob_ge_095": payload["summary"]["self_fraction_prob_ge_095"],
            "other_fraction_prob_ge_095": payload["summary"]["other_fraction_prob_ge_095"],
        }
        for seed, payload in zip(seeds, seed_payloads)
    }

    controls = {
        f"logit_bias_{b:.3f}": positive_control(seed_payloads, b, n_boot)
        for b in (0.025, 0.050, 0.100, 0.200)
    }
    return {
        "base": base_stats,
        "mixed_pooled": pooled,
        "mixed_runs": run_summaries,
        "assay_positive_control": controls,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# v0.6 identity-controlled hard evaluation", "",
        "Primary construct: counterfactual history sensitivity under lexical-identical history text; SELF/OTHER share identical historical text and differ only by an identity-assignment header.", "",
    ]
    for arch, d in result["architectures"].items():
        lines += [f"## {arch}", "", "### Mixed adapters — hierarchical pooled", "",
                  "| metric | mean | 95% CI |", "|---|---:|---:|"]
        for metric in METRICS:
            s = d["mixed_pooled"][metric]
            lines.append(f"| {metric} | {s['mean']:+.6f} | [{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}] |")
        eq = d["mixed_pooled"]["delta_prob"]["equivalence"]
        lines += ["", "Probability-scale equivalence:"]
        for k, v in eq.items():
            lines.append(f"- {k}: **{v['equivalent']}**")
        lines += ["", "### Base model", "", "| metric | mean | 95% CI |", "|---|---:|---:|"]
        for metric in METRICS:
            s = d["base"][metric]
            lines.append(f"| {metric} | {s['mean']:+.6f} | [{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}] |")
        lines += ["", "### Positive-control assay sensitivity", ""]
        for bias, block in d["assay_positive_control"].items():
            s = block["delta_sensitivity"]
            lines.append(f"- {bias}: Δ_sensitivity={s['mean']:+.4f}, CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}], detected={s['detected_positive']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="Contains qwen/ and mistral/ result directories")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    seeds = tuple(args.seeds)

    result = {
        "benchmark": "locked_v06_identity_controlled",
        "inference": "crossed hierarchical bootstrap over training seeds and latent item IDs",
        "seeds": list(seeds),
        "probability_equivalence_ropes": list(ROPE_PROB),
        "architectures": {
            "qwen4b": architecture_summary(args.root / "qwen", seeds, args.bootstrap),
            "mistral7b": architecture_summary(args.root / "mistral", seeds, args.bootstrap),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
