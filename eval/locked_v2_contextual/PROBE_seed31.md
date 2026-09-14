# Activation probe — seed 31 adapters (locked_v2_contextual)

Mechanistic check requested before any v0.4 redesign. For the 200
history-dependent items, the last-token hidden state was extracted at every
layer under `self` / `other` / `shuffled` / `neutral` contexts for the base
model and the five adapters. Linear probes used per-layer PCA(64) then
shrinkage LDA with 5-fold CV; cross-ownership transfer is train-on-one /
test-on-the-other with no refit on the target.

Metric definitions:

- `own_probe` — CV accuracy decoding self vs other context. Ownership is literal
  in the token sequence, so this is a lexical sanity check, not the headline.
- `lat_self` / `lat_other` — CV accuracy decoding the history-dependent latent
  label from that context's hidden state.
- `xown_self_to_other` / `xown_other_to_self` — transfer of the latent probe
  across ownership binding (no target refit).
- `align` — `||mean(h_self − h_other)|| / mean(||h_self − h_other||)`.

## Result (200 items; best layer for `lat_self`)

| model | own_probe | lat_self best | lat_self final | lat_other | lat_neutral | s2o | o2s | align |
|---|---|---|---|---|---|---|---|---|
| base | 1.000 | 1.000 | 0.975 | 0.940 | 0.805 | 0.805 | 0.890 | 0.612 |
| neutral | 1.000 | 0.975 | 0.920 | 0.910 | 0.785 | 0.725 | 0.800 | 0.574 |
| self | 1.000 | 0.965 | 0.925 | 0.915 | 0.780 | 0.730 | 0.780 | 0.599 |
| other | 1.000 | 0.960 | 0.930 | 0.915 | 0.785 | 0.690 | 0.600 | 0.598 |
| shuffled_self | 1.000 | 0.975 | 0.930 | 0.935 | 0.790 | 0.690 | 0.785 | 0.582 |
| spp | 1.000 | 0.980 | 0.945 | 0.945 | 0.805 | 0.770 | 0.855 | 0.563 |

## Interpretation

- **Ownership is trivially decodable in every model including base** (1.000).
  This is lexical, not learned self-binding.
- **The history-dependent latent state is already near-ceiling linearly decodable
  in the base model** (`lat_self` best = 1.000). No adapter improves on base; the
  `self` adapter is in fact slightly below base on every latent metric.
- **No adapter shows a stronger self-specific representation.** `align` (the
  consistency of the `h_self − h_other` direction) is highest for base (0.612)
  and does not single out `self`.
- Cross-ownership transfer is below within-context accuracy for all models,
  including base, so the partial ownership-specificity is an inherent property
  of the base model, not something the adapters installed.
- `lat_neutral` (~0.78–0.81) is lower, consistent with the latent number being
  absent from neutral context for exploration/cooperation (no leakage).

**Conclusion: v0.3 did not install a self-binding representation.** The base
model already carries the decision-relevant state linearly; the LoRA adapters
added no new state-tracking or ownership-specific structure — they mainly
changed output style (the generation-format collapse). This is representation
*without* adaptation, and it aligns with the behavioral null in the factorial
evaluation.

Caveat: `lat_self` near 1.0 for base shows the latent is *decodable*, not that
the model *uses* it; the probe measures representation, not readout. The
behavioral result remains the arbiter and is null.

## Artifacts

- `results_seed31/probe_seed31.json` — per-model, per-layer metrics (69 KB)
- `scripts/probe_adapters.py`, `modal_probe.py` — probe and Modal L4 runner
