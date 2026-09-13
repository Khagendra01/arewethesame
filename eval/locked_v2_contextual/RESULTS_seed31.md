# locked_v2_contextual — factorial result (seed 31 adapters)

Models `{base, neutral, self, other, shuffled_self, spp}` × contexts
`{neutral, self, other, shuffled, spp}` over 350 items. Primary metric is the
option-letter log-likelihood `E[P(correct)]`; hard accuracy is reported
alongside. Bootstrap CIs resample the 350 items (2000 draws).

## Matrix — E[P(correct)]

| model | neutral | self | other | shuffled | spp |
|---|---|---|---|---|---|
| base | 0.680 | 0.680 | 0.680 | 0.680 | 0.675 |
| neutral | 0.678 | 0.678 | 0.676 | 0.671 | 0.674 |
| self | 0.670 | 0.677 | 0.676 | 0.670 | 0.672 |
| other | 0.670 | 0.643 | 0.645 | 0.649 | 0.668 |
| shuffled_self | 0.648 | 0.645 | 0.654 | 0.646 | 0.650 |
| spp | 0.679 | 0.677 | 0.675 | 0.678 | 0.641 |

## Preregistered interactions

| statistic | E[P(correct)] | 95% CI | accuracy | 95% CI |
|---|---|---|---|---|
| Δ_bind | **+0.0031** | [−0.0044, +0.0104] | +0.0057 | [+0.0000, +0.0143] |
| Δ_coherence | **+0.0127** | [−0.0001, +0.0251] | −0.0057 | [−0.0314, +0.0171] |

- `Δ_bind = [P_self(self) − P_self(other)] − [P_other(self) − P_other(other)]`
- `Δ_coherence = [P_self(self) − P_self(shuffled)] − [P_other(self) − P_other(shuffled)]`

**Primary conclusion: no evidence of learned self-binding.** Δ_bind is null on
the primary metric. Δ_coherence is a weak positive blip whose CI lower bound
touches zero and which does not replicate in the accuracy metric. The pilot does
not support the claim that autobiographical training creates a special
self-binding mechanism.

## Per-family notes

- Control families (`belief_revision`, `trust`, `persistence`) saturate at
  P≈1.0 for every model and context.
- Threshold families carry the signal.
- The `other` model is markedly degraded by first-person contexts on
  `resource_allocation`: 0.594 (neutral) → 0.411 (self) / 0.420 (other). This is
  a model×context interaction, but opposite in sign to the self-binding
  hypothesis.
- The `self` model is the only one that improves on `exploration` when given its
  own context: 0.435 (neutral) → 0.492 (self). It is a small single-family
  effect, not a global interaction.
- The `spp` model scores lowest on its own `spp` context (aggregate 0.641).

## Secondary observation — generation format collapse

Exact-letter output on the 30-item greedy subset:

| model | clean letters |
|---|---|
| base | 120/120 |
| spp | 62/120 |
| neutral | 47/120 |
| shuffled_self | 30/120 |
| self | 17/120 |
| other | 16/120 |

The adapters largely reproduce their trained free-text decision style instead of
the requested letter. Samples in `results_seed31/generation_samples.json`.

## Limitations

- 350 items, single seed; three of seven families are at ceiling.
- Context variants differ in whether a history is present at all, so
  `neutral` cells conflate "no history" with "no binding".
- Adapters consistently fail the answer-format instruction, which may itself
  absorb model capacity.
- The `self`/`other` adapters share an identical answer-supervision signal, so a
  null Δ_bind is the result matched supervision predicts; the pilot cannot rule
  out effects that this instrument and dose are too weak to detect.

## Artifacts

- `items.jsonl` — frozen 350 items
- `results_seed31/summary.json` — matrices, interactions, per-family breakdown
- `results_seed31/generation_samples.json` — greedy generation samples
- Full per-item records (3 MB) remain on the Modal volume
  `arewethesame-pilot-seed31: eval_locked_v2/factorial.json`
