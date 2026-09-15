# Preregistration — Extension 3: Mistral-7B architecture replication

Model: `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0, 7B params).
Training: same v0.5 mixed corpus (2,800 items), same QLoRA recipe (NF4,
rank 16, 3 epochs, seed ∈ {31, 42, 73, 128, 256}), no hyperparameter tuning.
Only the mixed adapter is trained (the preregistered primary).

## Primary hypothesis

H0: Δ_within = 0 on both ID and OOD held-out sets.

Support for self-specific representation requires Δ_within > 0 with 95% CI
excluding 0 on **both** ID and OOD simultaneously.

## Statistics

For each seed s ∈ {31, 42, 73, 128, 256}:

```
Delta_within(s) = E[P_mixed(correct | self context)] - E[P_mixed(correct | other context)]
```

Reported separately for:
- ID: v0.5 held-out test (268 items, same generator policy)
- OOD: held-out templates (700 items, new families + new wording)

Pooled estimate: mean across seeds, CI from SD × 1.96.

## Sanity check

```
history_use = E[P(correct | coherent)] - E[P(correct | shuffled)]
```

If history_use ≈ 0, the null self-effect is uninterpretable (the model
didn't learn the task).

## Decision rules

- If Δ_within ≈ 0 on ID **and** OOD: strengthens the "no first-person
  privilege" conclusion across architectures.
- If Δ_within > 0 on OOD but not ID: architecture-specific generalization
  effect — interpret as mixed.
- If Δ_within > 0 on both: first-person privilege generalizes — interpret as
  positive.

## Limitations noted in advance

- 7B vs 4B capacity difference is acceptable because the comparison is
  within-Mistral, not cross-model.
- Single training recipe (no per-model tuning).
- Same frozen datasets as Qwen experiments.
