# Preregistration — v0.5 self-specificity test

Corpus: `experiments/assistant_v05_seed202` (2,800 matched history-dependent
items, 7 families). Conditions: `self`, `other`, `mixed`, `neutral`, `spp`.
Evaluation: held-out `test` instances (268 items), same generator policy,
contexts `neutral` / `self` / `other` / `shuffled` / `spp`.

Metric: option-letter log-likelihood, `E[P(correct)]`. Bootstrap CIs resample
items (2,000 draws). Significance threshold 95% (two-sided).

## Primary hypothesis

The `mixed` adapter chooses the binding id est (self / other / neutral) at
random per item during training while the target depends only on the latent
history. It therefore has equal training exposure to self and other bindings.
Any systematic self > other advantage at evaluation cannot be binding
familiarity.

**Primary statistic** (within a single model, so no cross-adapter contrast):

```
Delta_within = E[P_mixed(correct | self context)] − E[P_mixed(correct | other context)]
```

**H0:** `Delta_within = 0`. **H1:** `Delta_within > 0`.

Support for a self-specific representation requires the primary 95% CI to
exclude 0 on the positive side.

## Secondary (exploratory) statistics

- `Delta_asym = a_self − a_other`, where `a_self` / `a_other` are the
  own-binding advantages of the separately trained `self` / `other` adapters.
  This is the corrected version of the v0.4 `Delta_bind`.
- `Delta_bind = a_self + a_other` (reported for continuity; expected to be
  positive from binding familiarity alone, so not interpreted as self-specific).
- `Delta_coherence = [P_self(self) − P_self(shuffled)] − [P_other(self) − P_other(shuffled)]`.
- History-use sanity check: the coherent-vs-shuffled drop for each adapter.

## Decision rule

- Confirm self-specificity only if the **primary** CI excludes 0 (>0).
- Secondaries are reported as exploratory and do not by themselves establish
  the claim.
- Report all conditions and all contexts; no selective reporting.

## Limitations carried forward

Single seed; within-generator templates (held-out instances, not held-out
templates); the `mixed` adapter is trained on coherent history only in the
sense that the binding matches the history it carries.
