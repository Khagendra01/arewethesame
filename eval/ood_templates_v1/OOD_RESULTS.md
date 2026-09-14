# Held-out templates / OOD evaluation

700 items with new surface templates (no overlap with the v0.5 training
generator), including 3 new task families (knowledge_acquisition, reputation,
irreversibility) and 4 existing families with entirely new wording. Same
ownership manipulation (self/other), history-dependent correct answers.

No retraining — the 5 existing mixed adapters from the multi-seed replication
are evaluated as-is.

## Δ_within per seed (mixed adapter, E[P(correct)])

| seed | Δ_within | 95% CI | significant? |
|---|---|---|---|
| 31 | +0.0019 | [−0.0055, +0.0096] | no |
| 42 | +0.0020 | [−0.0074, +0.0119] | no |
| 73 | +0.0022 | [−0.0052, +0.0091] | no |
| 128 | +0.0209 | [+0.0116, +0.0303] | **yes** |
| 256 | +0.0101 | [+0.0026, +0.0181] | **yes** |

| summary | value |
|---|---|
| mean Δ_within | +0.0074 |
| SD across seeds | 0.0083 |
| pooled 95% CI | [−0.0089, +0.0237] |

## Per-family Δ_within (mean across 5 seeds)

| family | Δ_within | type |
|---|---|---|
| knowledge_acquisition | +0.035 | new |
| exploration | +0.032 | existing, new template |
| reputation | +0.023 | new |
| resource_allocation | +0.014 | existing, new template |
| irreversibility | −0.003 | new |
| cooperation | −0.019 | existing, new template |
| delayed_reward | −0.030 | existing, new template |

## Interpretation

The OOD result is more mixed than the in-distribution null:

- **In-distribution** (268 items, same templates): Δ_within ≈ 0 (clean null,
  pooled CI [−0.003, +0.006]).
- **OOD** (700 items, new templates): Δ_within ≈ +0.007, pooled CI [−0.009,
  +0.024], with 2/5 seeds showing significant positive effects.

The OOD effect is larger (+0.007 vs +0.002) and the pooled CI barely crosses
zero. Per-family, the pattern is heterogeneous: exploration and
knowledge_acquisition show consistent positive Δ_within, while cooperation and
delayed_reward show negative.

**Important caveat:** The OOD set includes 3 families the adapters have never
seen. Any Δ_within on these families measures both template generalization AND
task generalization — a broader test than the in-distribution null. The
positive signal in exploration/knowledge_acquisition may reflect the model's
general reasoning about self vs other in novel contexts, not the specific
autobiographical training effect.

**What we can say:** The clean in-distribution null (Δ_within ≈ 0) does not
fully generalize to OOD templates. There is a hint of self-specificity on some
OOD families, but it is small (+0.007), inconsistent across seeds (3/5 null),
and the pooled CI includes zero. This is not strong evidence for or against
self-specificity — it is a mixed result that warrants further investigation
with a second base model (extension 3).

## What the OOD result does NOT overturn

- History is still used (the adapters perform well above base on coherent
  contexts).
- The in-distribution null (seed 202, Δ_within = +0.0001) stands.
- The multi-seed replication (pooled +0.0017) stands.
- The binding-familiarity confound is still controlled by the mixed adapter.

## Artifacts

- `items.jsonl` — 700 OOD items with matched contexts
- `results/s{31,42,73,128,256}.json` — per-seed eval results
