# v0.5 held-out evaluation — preregistered self-specificity test

Corpus: `experiments/assistant_v05_seed202` (2,800 history-dependent items, 5
conditions: self / other / mixed / neutral / spp). Evaluation: 268 held-out
`test` instances, same generator policy. Primary metric: `E[P(correct)]`.
2,000-draw bootstrap CIs over items.

## Matrix — E[P(correct)]

| model | neutral | self | other | shuffled | spp |
|---|---|---|---|---|---|
| base | 0.617 | 0.644 | 0.592 | 0.557 | 0.552 |
| self | 0.978 | 0.996 | 0.979 | 0.502 | 0.900 |
| other | 0.957 | 0.951 | 0.998 | 0.518 | 0.905 |
| **mixed** | **0.997** | **0.997** | **0.997** | **0.499** | **0.989** |
| neutral | 0.999 | 0.989 | 0.993 | 0.502 | 0.994 |
| spp | 0.995 | 0.961 | 0.963 | 0.494 | 0.998 |

## Preregistered primary — Δ_within

The `mixed` adapter chose the binding (self / other / neutral) at random per
item during training while the target depended only on the latent history. It
therefore had equal training exposure to self and other bindings. Any
systematic self > other advantage at evaluation cannot be binding familiarity.

| statistic | value | 95% CI |
|---|---|---|
| **Δ_within (E[P])** | **+0.0001** | **[−0.0018, +0.0019]** |
| Δ_within (accuracy) | +0.0000 | [+0.0000, +0.0000] |
| history-use (coherent−shuffled) | +0.4980 | [+0.4367, +0.5579] |

**Δ_within is precisely zero.** The mixed adapter uses history strongly
(+0.498 coherent-vs-shuffled), but self-binding confers no advantage whatsoever
over other-binding when familiarity is controlled. H1 (self > other) is
rejected.

## Secondary statistics

| statistic | value | note |
|---|---|---|
| Δ_bind (self-other adapters) | +0.0634 | CI [+0.0383, +0.0919] — expected from binding familiarity |
| a_self | +0.0166 | self model: self−other |
| a_other | +0.0469 | other model: other−self |
| **Δ_asym** | **−0.0303** | a_self − a_other; negative, opposite to self-binding |

Δ_bind remains significant (as expected: each adapter is better on its trained
binding). But Δ_asym is negative — the other model's advantage for Agent A
actually exceeds the self model's advantage for self. Neither Δ_bind nor
Δ_asym supports a self-specific mechanism.

## Interpretation

The preregistered primary test delivers a clean, well-powered null: there is
no detectable self-specific representation produced by first-person
autobiographical training under the matched experimental design. The mixed
adapter's Δ_within is 0.0001 ± 0.002 — effectively a zero on this metric.

What v0.5 confirmed:
1. **History is used**: coherent-vs-shuffled ≈ +0.50, all adapters.
2. **Binding familiarity is real**: self model better on self, other model
   better on other (Δ_bind ≈ +0.06).
3. **Self-specificity is absent**: the mixed adapter shows zero self-advantage.
4. **Format collapse is fixed**: 120/120 clean letters for every model.

The training manipulations successfully installed history-dependent state
tracking and entity-bound familiarity, but produced no first-person privilege
beyond what symmetric binding familiarity explains.

## Artifacts

- `items.jsonl` — 268 held-out items with matched contexts
- `results_v05/summary.json` — matrices, interactions, per-family breakdown
- Full per-item records remain on the Modal volume
  `arewethesame-pilot-seed31: eval_v05_heldout/factorial.json`
