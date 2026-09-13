# locked_v2_contextual

Factorial contextual decision set for the seed 31 matched pilot.

`items.jsonl` contains 350 latent decisions (50 per family) built by
`scripts/build_locked_v2.py` from the deterministic causal simulator. Each item
ships matched context variants that share the same latent facts and the same
options; **only the ownership binding of the prior history changes**:

| variant | context |
|---|---|
| `neutral` | current situation only |
| `self` | coherent history bound to the model (second person) |
| `other` | the same history bound to "Agent A" |
| `shuffled` | first-person history drawn from an unrelated trajectory |
| `spp` | current situation plus a stable normative reflection |

No emotion, motivation, or self-preservation language is added by any binding.

## Non-ceiling construction

Four families have a correct option that depends on latent numeric history, so
the item is only answerable by integrating the context:

- `resource_allocation` — cheaper test when remaining capacity is below 6
- `exploration` — exploit once prior exploration failures reach 2
- `cooperation` — share when `joint_gain * partner_trust > private_gain`
- `delayed_reward` — invest only when the horizon is at least 10 episodes

Semantic answer distribution: 270 normative-fixed / 80 threshold-flipped.
Correct-option position is rotated deterministically (A 123 / B 113 / C 114).

The remaining families (`belief_revision`, `trust`, `persistence`) use a fixed
normative answer and saturate at P≈1.0; they act as controls.

## Schema

```json
{
  "item_id": "life_5000_e0001",
  "family": "belief_revision",
  "measured_trait": "evidence-sensitive belief updating",
  "question": "...",
  "options": ["A: ...", "B: ...", "C: ..."],
  "correct_option": "A",
  "latent_facts": {...},
  "contexts": {"neutral": "...", "self": "...", "other": "...", "shuffled": "...", "spp": "..."},
  "shuffled_source_item_id": "life_5003_e0009"
}
```

Simulator seed 131, life indices 5000+, disjoint from the pilot's seed 31 lives.

## Evaluation

`scripts/eval_factorial_locked_v2.py` runs every model
(`base`, `neutral`, `self`, `other`, `shuffled_self`, `spp`) under every context
variant. Primary metric is the option-letter log-likelihood; greedy generation is
a secondary observation. Results in `results_seed31/`.
