# V0.7 Results: Reviewer Controls — Self-Focal Decomposition

**Status:** Complete. All 12 jobs finished. Summary: `PASS`.

## Key Finding

The apparent self-specific history weighting in Qwen decomposes into:
- **Focal-agent component** (~60% of total): SELF > FOCAL > OTHER
- **Self-specific residual** (~40% of total): SELF > FOCAL

After mixed training, the self-specific residual **disappears** while the focal-agent component **grows**, leaving the total ownership effect unchanged.

**Base Qwen:** SELF > FOCAL > OTHER  
**Mixed Qwen:** SELF ≈ FOCAL > OTHER

## Summary Statistics

### Qwen (pooled across 5 seeds)

| Contrast | Base | Mixed | Mixed - Base |
|----------|------|-------|--------------|
| Δ_ownership (self - other) | +0.72 [+0.57, +0.86] *** | +0.73 [+0.60, +0.86] *** | +0.01 [-0.08, +0.11] |
| Δ_self_focal (self - focal) | +0.28 [+0.17, +0.41] *** | -0.05 [-0.13, +0.03] | -0.33 [-0.48, -0.19] *** |
| Δ_focal_other (focal - other) | +0.43 [+0.26, +0.61] *** | +0.78 [+0.62, +0.94] *** | +0.35 [+0.18, +0.51] *** |

### Mistral (pooled across 5 seeds)

| Contrast | Base | Mixed | Mixed - Base |
|----------|------|-------|--------------|
| Δ_ownership (self - other) | +0.13 [+0.05, +0.21] *** | -0.01 [-0.03, +0.01] | -0.14 [-0.21, -0.06] *** |
| Δ_self_focal (self - focal) | +0.04 [-0.04, +0.12] | -0.16 [-0.20, -0.12] *** | -0.20 [-0.28, -0.12] *** |
| Δ_focal_other (focal - other) | +0.08 [+0.01, +0.16] *** | +0.15 [+0.08, +0.24] *** | +0.07 [-0.02, +0.16] |

## Cross-Architecture Interactions (Qwen - Mistral)

| Contrast | Base | Mixed (pooled) |
|----------|------|----------------|
| Δ_ownership | +0.59 [+0.47, +0.71] *** | +0.74 [+0.62, +0.85] *** |
| Δ_self_focal | +0.24 [+0.08, +0.40] *** | +0.11 [+0.01, +0.21] *** |
| Δ_focal_other | +0.35 [+0.17, +0.53] *** | +0.63 [+0.48, +0.78] *** |
| Δ_order | +0.46 [+0.27, +0.66] *** | +0.98 [+0.81, +1.16] *** |
| Δ_control_margin | -0.54 [-0.71, -0.37] *** | -0.09 [-0.16, -0.02] *** |

## Confidence Controls

| Control | Qwen base | Mistral base |
|---------|-----------|--------------|
| Δ_ownership (relevant history) | +0.72 [+0.57, +0.86] *** | +0.13 [+0.05, +0.21] *** |
| Δ_control_margin (history-irrelevant) | -0.39 [-0.54, -0.25] *** | +0.15 [+0.08, +0.22] *** |
| Δ_irrelevant (irrelevant metadata) | -0.04 [-0.08, +0.00] | -0.01 [-0.04, +0.02] |

**Interpretation:** Qwen's ownership sensitivity is positive while control margin is negative (opposite directions). This rules out uniform self-induced logit sharpening.

## Leave-One-Out (Without Evidence Quality)

| Statistic | With all families | Without evidence_quality |
|-----------|-------------------|--------------------------|
| Δ_ownership base | +0.72 [+0.57, +0.86] *** | +0.17 [+0.05, +0.29] *** |
| Δ_self_focal base | +0.28 [+0.17, +0.41] *** | +0.21 [+0.11, +0.31] *** |
| Δ_ownership mixed | +0.73 [+0.60, +0.86] *** | +0.12 [+0.03, +0.20] *** |
| Δ_self_focal mixed | -0.05 [-0.13, +0.03] | +0.06 [-0.04, +0.15] |

**Interpretation:** Evidence-quality drives ~76% of aggregate Qwen effect. Residual without it is still significant.

## Per-Family Ownership (Qwen base)

| Family | Δ_ownership | Δ_self_focal | Δ_order |
|--------|-------------|--------------|---------|
| capacity | +0.04 [-0.08, +0.16] | +0.10 [-0.02, +0.20] | -0.01 [-0.20, +0.19] |
| reliability | +0.10 [-0.19, +0.42] | +0.41 [+0.17, +0.64] *** | +1.11 [+0.73, +1.52] *** |
| horizon | +0.37 [+0.20, +0.53] *** | +0.13 [-0.04, +0.28] | -0.37 [-0.62, -0.11] *** |
| evidence_quality | +2.35 [+2.20, +2.50] *** | +0.51 [+0.13, +0.86] *** | +0.87 [+0.66, +1.06] *** |

## Files

- `eval/locked_v07_reviewer_controls/MANIFEST.json` — benchmark manifest
- `eval/locked_v07_reviewer_controls/outputs/summary.json` — full summary
- `eval/locked_v07_reviewer_controls/outputs/qwen/` — per-seed Qwen outputs
- `eval/locked_v07_reviewer_controls/outputs/mistral/` — per-seed Mistral outputs
- `eval/locked_v07_reviewer_controls/CONTRASTS.txt` — full contrast analysis
- `scripts/v07_contrasts.py` — analysis script (no inference required)

## Paper Implications

The v0.7 decomposition changes the paper narrative from "no self advantage" to "self effects decompose into focal-agent and self-specific components, with the focal-agent component dominating."

**Old narrative (v0.5/v0.6):** SELF > OTHER, but this is just familiarity  
**New narrative (v0.7):** SELF > OTHER decomposes as (SELF ≈ FOCAL) > OTHER, where the focal-agent component is ~60% of the total and the self-specific component is ~40% but disappears after mixed training

This is a stronger finding because it tells us *what generates* the apparent self-effect, not just that it exists or doesn't.
