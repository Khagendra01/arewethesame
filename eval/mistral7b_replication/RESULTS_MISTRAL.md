# Extension 3 — Mistral-7B architecture replication

Model: `mistralai/Mistral-7B-Instruct-v0.3`. Same v0.5 mixed corpus, same QLoRA
recipe, 5 seeds (mixed adapter only). Evaluated on the identical ID (268 items)
and OOD (700 items) sets as the Qwen experiments. Preregistered primary:
Δ_within.

No hyperparameter tuning; no retraining of other conditions.

## Δ_within per seed (mixed adapter, E[P(correct)])

### ID held-out

| seed | Δ_within | 95% CI |
|---|---|---|
| 31 | −0.0007 | [−0.0020, −0.0000] |
| 42 | +0.0000 | [−0.0008, +0.0009] |
| 73 | −0.0036 | [−0.0114, +0.0007] |
| 128 | +0.0002 | [+0.0001, +0.0003] |
| 256 | +0.0029 | [−0.0019, +0.0108] |

**ID pooled: −0.0002, SD 0.0023, CI [−0.0048, +0.0043]. Clean null.**

### OOD held-out templates

| seed | Δ_within | 95% CI |
|---|---|---|
| 31 | −0.0061 | [−0.0146, +0.0023] |
| 42 | −0.0003 | [−0.0130, +0.0130] |
| 73 | −0.0027 | [−0.0149, +0.0090] |
| 128 | −0.0055 | [−0.0150, +0.0043] |
| 256 | **+0.0343** | **[+0.0185, +0.0501]** |

**OOD pooled: +0.0040, SD 0.0171, CI [−0.0296, +0.0375].**

Only 1/5 seeds (256) shows a significant positive OOD effect. The pooled CI is
wide and includes zero.

## History use (sanity check, seed 31)

| set | base | mixed |
|---|---|---|
| ID | 0.555 / 0.562 | **0.994 / 0.995** |
| OOD | 0.367 / 0.392 | **0.651 / 0.657** |

History is strongly used on both sets (mixed >> base). The null self-effect is
not a failure to learn the task.

## Cross-architecture comparison

| architecture | ID Δ_within (pooled) | OOD Δ_within (pooled) |
|---|---|---|
| Qwen 4B | +0.0017, [−0.003, +0.006] | +0.0074, [−0.009, +0.024] |
| Mistral 7B | −0.0002, [−0.005, +0.004] | +0.0040, [−0.030, +0.038] |

Both architectures give a clean null on ID. Both give a small, inconsistent
OOD effect whose pooled CI includes zero (Qwen 2/5 significant, Mistral 1/5
significant). **No architecture shows a self-specific privilege under this
training paradigm.** The OOD hint is present in both but is small, inconsistent
across seeds, and inconsistent across families.

## Interpretation

The "no intrinsic first-person privilege" conclusion from Qwen replicates on a
different model family with ~2x parameters. The OOD signal remains mixed in
both architectures: suggestive in isolated seeds, null in the pooled estimate.
This is not evidence for a self mechanism; it is evidence that any
self-specific effect, if it exists, is below the detection threshold of this
instrument and inconsistent across seeds, templates, and architectures.

## Artifacts

- `results/s{S}_{id,ood}.json` — per-seed per-set eval results (10 files)
