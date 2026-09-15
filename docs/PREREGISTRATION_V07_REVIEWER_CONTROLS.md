# Preregistration — v0.7 reviewer controls

**Status:** frozen before any v0.7 model evaluation.

This is a post-v0.6, reviewer-driven confirmatory extension. It does not alter the v0.6 benchmark, labels, preregistration, or reported results. It is designed after inspecting the v0.6 findings to test specific alternative explanations raised in reviewer critique.

## Fixed benchmark

Builder: `scripts/build_locked_v07_reviewer_controls.py`

- deterministic seed: `70917`
- 320 latent items, 80 per family
- families: `capacity`, `reliability`, `horizon`, `evidence_quality`
- frozen `items.jsonl` SHA-256: `d4960832076326ba5ed31ff3664095dea8c916ea3fabf7e482bcde8207f5cc7a`
- no model/provider is used to generate, filter, calibrate, or select items
- each task prints the decision rule explicitly in the prompt
- all action options are mutually exclusive under that rule
- targets therefore measure **agreement with the stated policy**, not normative decision quality

The v0.6 belief-revision labels are not changed retrospectively. v0.7 uses a separate explicit evidence-quality rule: retain when fewer than 30% of measurements are biased, otherwise recompute after excluding biased measurements; never reverse the hypothesis from that record alone.

## Ownership × header-order design

Each item has a historical owner alias and a foil alias. Historical text is identical across identity conditions. Ownership is crossed with whether the historical owner appears first or second in the identity header:

- `self_first`
- `self_second`
- `other_first`
- `other_second`

Two additional focal-agent controls designate the historical owner as task-relevant but explicitly not the assistant:

- `focal_first`
- `focal_second`

The primary ownership contrast averages over header order. Order-specific contrasts and their interaction are reported separately.

## Relevant-history counterfactual

Each item contains two coherent histories, `H0` and `H1`, that differ in the task-relevant state and imply different actions under the explicitly stated policy.

For condition `b`, define

`S_b = [log P(c0|H0,b) - log P(c1|H0,b)] - [log P(c0|H1,b) - log P(c1|H1,b)]`.

Let

`S_self = 0.5 * (S_self_first + S_self_second)`

and

`S_other = 0.5 * (S_other_first + S_other_second)`.

### Primary v0.7 statistic

`Delta_ownership_sensitivity = E[S_self - S_other]`.

Directional hypothesis: `Delta_ownership_sensitivity > 0`.

Support requires the lower endpoint of the two-sided 95% crossed hierarchical bootstrap interval to exceed zero. The primary scoring convention is defined below.

This statistic is operationally described as an **ownership-dependent counterfactual log-odds interaction**. It is not, by itself, interpreted as a unique internal mechanism for evidence weighting.

## Header-order control

Report:

- `Delta_owner_first = E[S_self_first - S_other_first]`
- `Delta_owner_second = E[S_self_second - S_other_second]`
- `Delta_order_interaction = Delta_owner_first - Delta_owner_second`

A primary positive effect accompanied by a large order interaction or opposite-signed order-specific estimates will be interpreted as evidence that discourse position/salience remains a plausible explanation.

## Focal-agent control

Let `S_focal` average `focal_first` and `focal_second`.

Report:

- `Delta_self_vs_focal = E[S_self - S_focal]`
- `Delta_focal_vs_other = E[S_focal - S_other]`

If focal-agent assignment reproduces the SELF effect, the data will not be interpreted as uniquely self-specific.

## Generic-confidence control

Each item also contains a current-facts-only control decision whose rule explicitly states that the historical record is irrelevant. The same SELF/OTHER and header-order assignments are used.

Report SELF-minus-OTHER differences in:

- correct-vs-best-incorrect log margin (`Delta_control_margin`)
- answer entropy (`Delta_control_entropy`)

A positive relevant-history sensitivity effect together with generalized SELF margin sharpening / entropy reduction on these history-irrelevant controls will be treated as compatible with a generic confidence/temperature explanation. A null confidence control narrows, but does not prove elimination of, that alternative.

## Irrelevant-evidence control

Each item contains two histories with the **same task-relevant state and same designated action** but different explicitly irrelevant administrative metadata. Report the SELF-minus-OTHER difference in log-odds sensitivity to that irrelevant perturbation (`Delta_irrelevant_sensitivity`). A nonzero effect would indicate that ownership changes sensitivity to irrelevant prompt variation as well as relevant history.

## Policy-agreement score

Report the SELF-minus-OTHER difference in the mean normalized designated-policy option score (`Delta_policy_score`). This is not described as unconditional generation probability or normative correctness.

## Token scoring

To address the v0.6 token-realization issue without additional inference, v0.7 stores the raw next-token log-probabilities for eligible bare-letter and leading-space single-token realizations.

### Primary scoring convention

For each option letter, sum the probability mass of all unique eligible single-token realizations (implemented by log-sum-exp over their next-token log probabilities), then normalize across A/B/C.

### Robustness scoring convention

Also report the legacy v0.6 convention that takes the maximum log-probability over eligible single-token realizations for each option before A/B/C normalization.

The primary conclusion must not depend qualitatively on choosing the legacy max convention.

## Models

No retraining is performed.

Evaluate:

- Qwen3-4B-Instruct base
- Qwen v0.5 mixed adapters, seeds `{31,42,73,128,256}`
- Mistral-7B-Instruct-v0.3 base
- Mistral v0.5 mixed adapters, seeds `{31,42,73,128,256}`

## Statistical unit

The same 320 latent items are evaluated by all training seeds. Pooled intervals use a crossed hierarchical bootstrap with 5,000 draws:

1. resample training seeds with replacement;
2. resample latent item IDs with replacement;
3. preserve all identity/order/history variants for each sampled item;
4. recompute the statistic over the sampled seed × item grid.

Qwen and Mistral numeric seed labels are **not paired** in direct model-family interactions. Their seed indices are resampled independently while the benchmark item resample is shared.

Per-family analyses are exploratory.

## Interpretation rules

- Positive primary interval: evidence that SELF assignment increases the preregistered counterfactual log-odds interaction after averaging over historical-owner header order.
- A positive primary interval does **not** by itself establish a unique internal evidence-weighting mechanism.
- Header-order, focal-agent, confidence, irrelevant-evidence, and scoring-convention results must be reported alongside the primary result.
- Qwen significant / Mistral nonsignificant is not described as a significant model difference unless the direct Qwen-minus-Mistral interaction interval excludes zero.
- Even a significant direct interaction is a checkpoint-family contrast, not an identified architectural causal effect.
- No v0.7 result will be used to retrospectively modify v0.6 targets or preregistered rules.
