#!/usr/bin/env python3
"""
Compute all v0.7 contrasts from existing output JSONs.
No model inference required — reads per-item metrics from frozen outputs.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT = Path("eval/locked_v07_reviewer_controls/outputs")

def load_all():
    """Load all 12 output JSONs."""
    data = {}
    for arch in ["qwen", "mistral"]:
        data[arch] = {}
        for variant in ["base", "mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
            p = OUT / arch / f"{variant}.json"
            if p.exists():
                with open(p) as f:
                    data[arch][variant] = json.load(f)
    return data


def item_metrics(d, scoring="mass"):
    """Extract per-item metric dict keyed by item_id."""
    items = {}
    for item_id, item_data in d["items"].items():
        metrics = item_data[f"metrics_{scoring}"]
        items[item_id] = metrics
    return items


def pool_items(*item_dicts):
    """Pool per-item metrics across multiple dicts (e.g., multiple seeds).
    Returns dict of metric_name -> list of per-item values."""
    pooled = defaultdict(list)
    for d in item_dicts:
        for item_id, metrics in d.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    pooled[k].append(v)
    return dict(pooled)


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap mean CI."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


def bootstrap_diff_ci(vals_a, vals_b, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap CI for mean(a) - mean(b), paired on items."""
    rng = np.random.RandomState(seed)
    a = np.array(vals_a)
    b = np.array(vals_b)
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    n = len(a)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    boot_diffs = np.array(boot_diffs)
    mean_diff = float(np.mean(a) - np.mean(b))
    lo = float(np.percentile(boot_diffs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_diffs, (1 + ci) / 2 * 100))
    return mean_diff, lo, hi


def sig_marker(ci_lo, ci_hi):
    if ci_lo > 0 or ci_hi < 0:
        return "***"
    return ""


def format_result(mean, lo, hi):
    return f"{mean:+.4f}  [{lo:+.4f}, {hi:+.4f}] {sig_marker(lo, hi)}"


def main():
    data = load_all()

    # ---- Aggregate metrics from summary for reference ----
    print("=" * 90)
    print("V0.7 CONTRAST ANALYSIS — FROM EXISTING DATA (NO NEW INFERENCE)")
    print("=" * 90)

    # =========================================================================
    # 1. MIXED-BASE CONTRASTS (paired on items)
    # =========================================================================
    print("\n" + "=" * 90)
    print("1. MIXED-BASE CONTRASTS (Δ_mixed - Δ_base)")
    print("   Paired bootstrap over 320 shared items × 5 seeds")
    print("=" * 90)

    key_metrics = [
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
    ]

    for arch in ["qwen", "mistral"]:
        print(f"\n--- {arch.upper()} ---")
        base_items = item_metrics(data[arch]["base"])

        # Collect all mixed-seed item metrics
        mixed_seeds = {}
        for seed_name in ["mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
            if seed_name in data[arch]:
                mixed_seeds[seed_name] = item_metrics(data[arch][seed_name])

        # For each metric, compute pooled mixed mean vs base mean, paired on items
        for metric in key_metrics:
            base_vals = []
            mixed_vals = []
            # Pool across seeds for each item
            item_ids = sorted(base_items.keys())
            for item_id in item_ids:
                base_v = base_items[item_id].get(metric)
                if base_v is None or not isinstance(base_v, (int, float)):
                    continue
                # Average across mixed seeds for this item
                mixed_vs = []
                for sn, sd in mixed_seeds.items():
                    if item_id in sd and metric in sd[item_id]:
                        mv = sd[item_id][metric]
                        if isinstance(mv, (int, float)):
                            mixed_vs.append(mv)
                if mixed_vs:
                    base_vals.append(base_v)
                    mixed_vals.append(np.mean(mixed_vs))

            if len(base_vals) > 10:
                mean_diff, lo, hi = bootstrap_diff_ci(mixed_vals, base_vals)
                print(f"  {metric:42s} {format_result(mean_diff, lo, hi)}")
            else:
                print(f"  {metric:42s} [insufficient paired items]")

    # =========================================================================
    # 2. CROSS-ARCHITECTURE CHECKPOINT INTERACTIONS
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. CROSS-ARCHITECTURE INTERACTIONS (Qwen - Mistral)")
    print("   Paired on shared benchmark items")
    print("=" * 90)

    for variant_label, variant in [("base", "base"), ("mixed (pooled)", "mixed")]:
        print(f"\n--- {variant_label} ---")
        qwen_base = item_metrics(data["qwen"]["base"])
        mistral_base = item_metrics(data["mistral"]["base"])

        if variant == "mixed":
            # Pool across seeds
            qwen_mixed_seeds = {}
            mistral_mixed_seeds = {}
            for sn in ["mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
                if sn in data["qwen"]:
                    qwen_mixed_seeds[sn] = item_metrics(data["qwen"][sn])
                if sn in data["mistral"]:
                    mistral_mixed_seeds[sn] = item_metrics(data["mistral"][sn])

        for metric in key_metrics:
            qwen_vals = []
            mistral_vals = []
            item_ids = sorted(set(qwen_base.keys()) & set(mistral_base.keys()))

            for item_id in item_ids:
                if variant == "base":
                    qv = qwen_base[item_id].get(metric)
                    mv = mistral_base[item_id].get(metric)
                    if isinstance(qv, (int, float)) and isinstance(mv, (int, float)):
                        qwen_vals.append(qv)
                        mistral_vals.append(mv)
                else:
                    # Pool mixed seeds
                    q_vs = []
                    m_vs = []
                    for sn in ["mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
                        if sn in qwen_mixed_seeds and item_id in qwen_mixed_seeds[sn]:
                            v = qwen_mixed_seeds[sn][item_id].get(metric)
                            if isinstance(v, (int, float)):
                                q_vs.append(v)
                        if sn in mistral_mixed_seeds and item_id in mistral_mixed_seeds[sn]:
                            v = mistral_mixed_seeds[sn][item_id].get(metric)
                            if isinstance(v, (int, float)):
                                m_vs.append(v)
                    if q_vs and m_vs:
                        qwen_vals.append(np.mean(q_vs))
                        mistral_vals.append(np.mean(m_vs))

            if len(qwen_vals) > 10:
                mean_diff, lo, hi = bootstrap_diff_ci(qwen_vals, mistral_vals)
                print(f"  {metric:42s} {format_result(mean_diff, lo, hi)}")
            else:
                print(f"  {metric:42s} [insufficient paired items]")

    # =========================================================================
    # 3. HEADER-ORDER INTERACTION & PER-FAMILY ESTIMATES
    # =========================================================================
    print("\n" + "=" * 90)
    print("3. HEADER-ORDER INTERACTION & PER-FAMILY ESTIMATES")
    print("=" * 90)

    for arch in ["qwen", "mistral"]:
        print(f"\n--- {arch.upper()} ---")
        base_d = data[arch]["base"]

        # Summary-level header-order interaction
        ho = base_d["summary_mass"]["ownership_by_order_interaction"]
        print(f"  Base header-order interaction: {format_result(ho['mean'], ho['ci95'][0], ho['ci95'][1])}")

        # Order-specific contrasts
        of = base_d["summary_mass"]["delta_owner_first"]
        os_ = base_d["summary_mass"]["delta_owner_second"]
        print(f"    delta_owner_first:            {format_result(of['mean'], of['ci95'][0], of['ci95'][1])}")
        print(f"    delta_owner_second:           {format_result(os_['mean'], os_['ci95'][0], os_['ci95'][1])}")

        # Per-family
        print(f"\n  Per-family (base, mass scoring):")
        for fam in ["capacity", "reliability", "horizon", "evidence_quality"]:
            fam_d = base_d["per_family_mass"][fam]
            ownership = fam_d["delta_ownership_sensitivity"]
            self_focal = fam_d["delta_self_vs_focal_sensitivity"]
            focal_other = fam_d["delta_focal_vs_other_sensitivity"]
            ctrl = fam_d["delta_control_margin"]
            irr = fam_d["delta_irrelevant_sensitivity"]
            ho_fam = fam_d["ownership_by_order_interaction"]
            print(f"    {fam:20s} ownership={format_result(ownership['mean'], ownership['ci95'][0], ownership['ci95'][1])}")
            print(f"    {'':20s} self_vs_focal={format_result(self_focal['mean'], self_focal['ci95'][0], self_focal['ci95'][1])}")
            print(f"    {'':20s} focal_vs_other={format_result(focal_other['mean'], focal_other['ci95'][0], focal_other['ci95'][1])}")
            print(f"    {'':20s} header_interaction={format_result(ho_fam['mean'], ho_fam['ci95'][0], ho_fam['ci95'][1])}")
            print(f"    {'':20s} control_margin={format_result(ctrl['mean'], ctrl['ci95'][0], ctrl['ci95'][1])}")
            print(f"    {'':20s} irrelevant={format_result(irr['mean'], irr['ci95'][0], irr['ci95'][1])}")

    # =========================================================================
    # 4. LEAVE-ONE-OUT: DELTA WITHOUT EVIDENCE_QUALITY
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. LEAVE-ONE-OUT: AGGREGATE delta WITHOUT evidence_quality FAMILY")
    print("=" * 90)

    for arch in ["qwen", "mistral"]:
        print(f"\n--- {arch.upper()} ---")
        for variant_label, variant in [("base", "base"), ("mixed (pooled)", "mixed")]:
            if variant == "base":
                item_d = item_metrics(data[arch]["base"])
            else:
                # Pool mixed seeds
                pooled_seeds = {}
                for sn in ["mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
                    if sn in data[arch]:
                        pooled_seeds[sn] = item_metrics(data[arch][sn])

            # Get family assignment from base
            base_items = data[arch]["base"]["items"]
            families = {item_id: item_data["family"] for item_id, item_data in base_items.items()}

            for metric in ["delta_ownership_sensitivity", "delta_self_vs_focal_sensitivity",
                           "delta_focal_vs_other_sensitivity", "delta_control_margin",
                           "delta_irrelevant_sensitivity", "delta_policy_score"]:
                vals = []
                for item_id in sorted(families.keys()):
                    if families[item_id] == "evidence_quality":
                        continue
                    if variant == "base":
                        v = item_d.get(item_id, {}).get(metric)
                        if isinstance(v, (int, float)):
                            vals.append(v)
                    else:
                        vs = []
                        for sn, sd in pooled_seeds.items():
                            if item_id in sd:
                                mv = sd[item_id].get(metric)
                                if isinstance(mv, (int, float)):
                                    vs.append(mv)
                        if vs:
                            vals.append(np.mean(vs))

                if len(vals) > 10:
                    mean, lo, hi = bootstrap_ci(vals)
                    print(f"  {variant_label:12s} {metric:42s} {format_result(mean, lo, hi)}")
                else:
                    print(f"  {variant_label:12s} {metric:42s} [insufficient items]")

    # =========================================================================
    # 5. SELF vs FOCAL DECOMPOSITION SUMMARY
    # =========================================================================
    print("\n" + "=" * 90)
    print("5. SELF-FOCAL DECOMPOSITION (THE KEY RESULT)")
    print("=" * 90)

    for arch in ["qwen", "mistral"]:
        print(f"\n--- {arch.upper()} ---")
        base_d = data[arch]["base"]

        s = base_d["summary_mass"]
        ownership = s["delta_ownership_sensitivity"]
        self_focal = s["delta_self_vs_focal_sensitivity"]
        focal_other = s["delta_focal_vs_other_sensitivity"]

        print(f"  Base:")
        print(f"    delta_ownership_sensitivity (S_self - S_other):  {format_result(ownership['mean'], ownership['ci95'][0], ownership['ci95'][1])}")
        print(f"    delta_self_vs_focal (S_self - S_focal):          {format_result(self_focal['mean'], self_focal['ci95'][0], self_focal['ci95'][1])}")
        print(f"    delta_focal_vs_other (S_focal - S_other):        {format_result(focal_other['mean'], focal_other['ci95'][0], focal_other['ci95'][1])}")

        # For mixed, pool across seeds
        pooled_seeds = {}
        for sn in ["mixed_s31", "mixed_s42", "mixed_s73", "mixed_s128", "mixed_s256"]:
            if sn in data[arch]:
                pooled_seeds[sn] = item_metrics(data[arch][sn])

        item_ids = sorted(data[arch]["base"]["items"].keys())
        for metric_name, metric_key in [
            ("ownership", "delta_ownership_sensitivity"),
            ("self_vs_focal", "delta_self_vs_focal_sensitivity"),
            ("focal_vs_other", "delta_focal_vs_other_sensitivity"),
        ]:
            vals = []
            for item_id in item_ids:
                vs = []
                for sn, sd in pooled_seeds.items():
                    if item_id in sd:
                        mv = sd[item_id].get(metric_key)
                        if isinstance(mv, (int, float)):
                            vs.append(mv)
                if vs:
                    vals.append(np.mean(vs))
            if len(vals) > 10:
                mean, lo, hi = bootstrap_ci(vals)
                print(f"  Mixed (pooled):")
                print(f"    delta_{metric_name:20s} {format_result(mean, lo, hi)}")

        # Decomposition check: does self ≈ focal in mixed?
        print(f"\n  Decomposition check:")
        print(f"    Base:   SELF > FOCAL > OTHER ? (ownership > self_focal > 0)")
        print(f"    Mixed:  SELF ≈ FOCAL > OTHER ? (self_focal → 0, ownership stays positive)")

    # =========================================================================
    # 6. CONFIDENCE OBJECTION: DIRECTION CHECK
    # =========================================================================
    print("\n" + "=" * 90)
    print("6. CONFIDENCE OBJECTION: DIRECTION CHECK")
    print("   Does SELF merely sharpen all logits, or is the effect specific?")
    print("=" * 90)

    for arch in ["qwen", "mistral"]:
        print(f"\n--- {arch.upper()} ---")
        s = data[arch]["base"]["summary_mass"]
        ownership = s["delta_ownership_sensitivity"]
        ctrl = s["delta_control_margin"]
        irr = s["delta_irrelevant_sensitivity"]

        print(f"  Relevant-history ownership sensitivity:  {format_result(ownership['mean'], ownership['ci95'][0], ownership['ci95'][1])}")
        print(f"  Current-facts-only control margin:        {format_result(ctrl['mean'], ctrl['ci95'][0], ctrl['ci95'][1])}")
        print(f"  Irrelevant-evidence sensitivity:          {format_result(irr['mean'], irr['ci95'][0], irr['ci95'][1])}")

        # Direction check
        own_dir = "positive" if ownership['mean'] > 0 else "negative"
        ctrl_dir = "positive" if ctrl['mean'] > 0 else "negative"
        irr_dir = "nonzero" if (irr['ci95'][0] > 0 or irr['ci95'][1] < 0) else "zero"

        if ownership['mean'] > 0 and ctrl['mean'] < 0:
            print(f"  → Ownership {own_dir}, control margin {ctrl_dir}: OPPOSITE directions")
            print(f"    → Rules out uniform SELF-induced confidence increase")
        elif ownership['mean'] > 0 and ctrl['mean'] > 0:
            print(f"  → Both positive: confidence scaling remains plausible")
        else:
            print(f"  → Ownership {own_dir}, control margin {ctrl_dir}")

        if irr_dir == "zero":
            print(f"  → Irrelevant sensitivity ≈ 0: SELF does not amplify arbitrary changes")
        else:
            print(f"  → Irrelevant sensitivity nonzero:SELF amplifies irrelevant changes too")


if __name__ == "__main__":
    main()
