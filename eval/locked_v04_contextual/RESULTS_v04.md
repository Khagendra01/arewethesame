# v0.4 held-out contextual evaluation

Adapters trained on `experiments/assistant_v04_seed101` (history-dependent,
neutral/self/other/spp, anchor data mixed into every run), evaluated on the
held-out `test` instances regenerated with the same generator policy
(`items.jsonl`, 129 items). Primary metric: option-letter log-likelihood.
Bootstrap CIs resample the 129 items (2000 draws).

Why not locked_v2: the earlier locked_v2 rules differ from the v0.4 policy
(resource threshold 6 vs 4, cooperation rule, three fixed-answer families), so
its labels are not a valid target set for v0.4. The locked_v2 numbers are not
reported here.

## Matrix — E[P(correct)]

| model | neutral | self | other | shuffled | spp |
|---|---|---|---|---|---|
| base | 0.634 | 0.657 | 0.661 | 0.604 | 0.541 |
| neutral | 0.980 | 0.970 | 0.968 | 0.595 | 0.990 |
| self | 0.924 | **0.988** | 0.956 | 0.589 | 0.957 |
| other | 0.962 | 0.957 | **0.985** | 0.575 | 0.859 |
| spp | 0.980 | 0.935 | 0.947 | 0.612 | 0.992 |

## Preregistered interactions (E[P(correct)])

| statistic | value | 95% CI |
|---|---|---|
| Δ_bind | **+0.0588** | [+0.0210, +0.1059] |
| Δ_coherence | +0.0159 | [−0.0262, +0.0586] |

Δ_bind is significantly positive on the primary metric; Δ_coherence is null.

## Decomposition — and why Δ_bind is not yet evidence of self-binding

Define per-model own-binding advantage:

- `a_self  = P_selfmodel(self)  − P_selfmodel(other)`
- `a_other = P_othermodel(other) − P_othermodel(self)`

| quantity | value | 95% CI |
|---|---|---|
| a_self | +0.0315 | [+0.0045, +0.0632] |
| a_other | +0.0274 | [−0.0001, +0.0580] |
| **Δ_asym = a_self − a_other** | **+0.0041** | **[−0.0324, +0.0471]** |
| Δ_bind = a_self + a_other | +0.0588 | [+0.0210, +0.1059] |

Because Δ_bind is the **sum** of the two own-binding advantages, a positive
Δ_bind is exactly what you get when each LoRA is simply better calibrated to
the entity binding it was trained on (self saw "you/your"; other saw "Agent A").
That symmetric familiarity is trivial and not a self mechanism. The
self-specific asymmetry `Δ_asym` is null (+0.004). **v0.4 therefore produces
binding familiarity but no detectable self-specific representation gain.**

## What v0.4 did achieve

- **History is actually used.** Adapters reach 0.92–0.99 when the history is
  coherent and collapse to ~0.58–0.61 when it is shuffled, while base sits at
  chance (~0.64) in all contexts. The v0.3 failure mode (history irrelevant to
  the target) is gone.
- **Format collapse is fixed.** Exact-letter generation is 120/120 for every
  model on the 30-item subset (v0.3: self 17/120, other 16/120). The anchor data
  worked.
- Eval losses 1.7e-4–1.7e-3 (v0.3: ~1e-5), consistent with a genuinely harder,
  non-memorizable task.

## Limitations

- Single seed, 129 held-out items, one generator policy; same templates as
  training, so this is within-generator generalization, not cross-template
  transfer.
- The Δ_bind statistic as preregistered confounds own-binding familiarity with
  self-specificity. Future work should preregister `Δ_asym` (or include a
  shuffled-binding adapter that never saw either entity) as the primary test.
- `shuffled_self` was not trained (ill-defined under history-dependent matched
  targets); the shuffled context is used only as an on-the-fly coherence probe.

## Artifacts

- `items.jsonl` — 129 held-out items with matched contexts
- `results_v04/summary.json` — matrices, interactions, per-family breakdown
- `results_v04/generation_samples.json` — greedy generation samples
- Full per-item records remain on the Modal volume
  `arewethesame-pilot-seed31: eval_v04_heldout/factorial.json`
