# V0.7 Results: Reviewer Controls — Hierarchical Final

**Status:** Complete. All 12 jobs finished. Primary inference below uses the preregistered crossed bootstrap over training seeds and latent items for mixed-adapter quantities.

## Qwen decomposition

| Contrast | Base | Mixed | Mixed - Base |
|---|---:|---:|---:|
| ownership (SELF-OTHER) | +0.715 [+0.571,+0.861] | +0.730 [+0.576,+0.897] | +0.015 [-0.118,+0.149] |
| SELF-FOCAL | +0.285 [+0.166,+0.406] | -0.048 [-0.214,+0.159] | -0.333 [-0.544,-0.098] |
| FOCAL-OTHER | +0.431 [+0.250,+0.608] | +0.778 [+0.500,+1.057] | +0.347 [+0.049,+0.620] |
| header-order interaction | +0.402 [+0.251,+0.560] | +0.704 [+0.498,+0.934] | +0.303 [+0.060,+0.565] |

Base Qwen therefore contains both a SELF-over-FOCAL and a FOCAL-over-OTHER component. After SELF/OTHER/NEUTRAL mixed training, total ownership is unchanged, while the aggregate SELF-FOCAL component decreases significantly and FOCAL-OTHER increases significantly. FOCAL is evaluation-only and was not exposure-matched during training, so this decomposition is a behavioral control, not a pure causal isolation of semantic selfhood.

## Mistral

| Contrast | Base | Mixed | Mixed - Base |
|---|---:|---:|---:|
| ownership | +0.128 [+0.046,+0.209] | -0.008 [-0.087,+0.065] | -0.135 [-0.242,-0.033] |
| SELF-FOCAL | +0.044 [-0.035,+0.121] | -0.159 [-0.231,-0.084] | -0.203 [-0.301,-0.106] |
| FOCAL-OTHER | +0.084 [+0.010,+0.160] | +0.151 [+0.077,+0.236] | +0.068 [-0.040,+0.180] |

## Direct Qwen - Mistral interactions

| Contrast | Base | Mixed |
|---|---:|---:|
| ownership | +0.588 [+0.465,+0.708] | +0.737 [+0.583,+0.906] |
| SELF-FOCAL | +0.240 [+0.076,+0.399] | +0.111 [-0.072,+0.335] |
| FOCAL-OTHER | +0.347 [+0.174,+0.529] | +0.627 [+0.345,+0.920] |
| header-order interaction | +0.458 [+0.264,+0.653] | +0.981 [+0.653,+1.352] |

The mixed SELF-FOCAL checkpoint difference is not reliably different from zero. Qwen and Mistral differ along multiple dimensions, so these are checkpoint-family interactions rather than architectural causal effects.

## Controls and heterogeneity

- Qwen base relevant-history ownership is positive (+0.715), while the history-irrelevant control margin is negative (-0.391 [-0.542,-0.246]); mixed control margin is near zero (-0.071 [-0.146,+0.003]). This is evidence against the simplest uniform SELF-induced confidence-sharpening account, not proof against every confidence mechanism.
- Qwen irrelevant-metadata sensitivity is small: base -0.042 [-0.084,-0.000] and mixed -0.016 [-0.033,+0.001].
- Excluding `evidence_quality`, Qwen ownership is +0.170 [+0.049,+0.295] in base but +0.117 [-0.001,+0.228] in mixed. Thus evidence-quality is a disproportionate driver and the mixed residual outside that family is inconclusive.
- Qwen mixed SELF-FOCAL is heterogeneous by family: capacity +0.259 [+0.174,+0.367], evidence_quality -0.360 [-0.752,+0.106], horizon +0.159 [+0.038,+0.322], reliability -0.251 [-0.541,+0.073]. Aggregate near-zero SELF-FOCAL therefore does not imply every family is null.

## Authoritative artifacts

- `outputs/summary.json`: preregistered hierarchical aggregate from the v0.7 evaluator.
- `CONTRASTS_HIERARCHICAL.json`: corrected 10,000-draw hierarchical contrasts, including mixed-minus-base and leave-one-out analyses.
- `CONTRASTS.txt`: human-readable rendering of the corrected hierarchical contrasts.
- `scripts/v07_contrasts.py`: reproducible CPU-only analysis; no model inference.
