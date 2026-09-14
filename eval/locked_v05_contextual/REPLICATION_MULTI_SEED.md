# Multi-seed replication — Δ_within stability

Five independent mixed-adapter trains on the same v0.5 corpus (2,800 items),
same hyperparameters, different random seeds. Each evaluated on the identical
268-item held-out set. Primary metric: Δ_within = E[P(correct | self)] −
E[P(correct | other)] for the mixed adapter.

## Results

| seed | Δ_within (E[P]) | 95% CI | significant? |
|---|---|---|---|
| 31 | +0.0001 | [−0.0018, +0.0020] | no |
| 42 | +0.0044 | [−0.0004, +0.0102] | no |
| 73 | +0.0032 | [−0.0062, +0.0125] | no |
| 128 | +0.0024 | [−0.0011, +0.0074] | no |
| 256 | −0.0016 | [−0.0098, +0.0042] | no |

| summary | value |
|---|---|
| mean Δ_within | +0.0017 |
| SD across seeds | 0.0024 |
| pooled 95% CI | [−0.0030, +0.0064] |

All five CIs include zero. The mean effect (+0.0017) is well within noise (SD
0.0024). No seed shows a significant self-specific advantage.

## Interpretation

The null result from seed 202 (Δ_within = +0.0001) is stable: across 5 seeds,
the estimated self-specific effect ranges from −0.002 to +0.004, with a pooled
point estimate of +0.0017 and tight bounds [−0.003, +0.006]. First-person
autobiographical training does not produce a detectable self-binding advantage
under the matched experimental design, and this is not a seed artifact.

## Remaining limitations (per preregistration)

- Same generator templates (held-out instances, not held-out templates).
- Same base model family (Qwen 3-4B).

## Artifacts

- `results_replication/s{31,42,73,128,256}.json` — per-seed eval results
