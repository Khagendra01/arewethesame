# Dataset design

## Research question

Can a persistent autobiographical self-model become a computationally useful organizing variable in a language model, rather than merely producing first-person style?

## Core principle

Each latent event is rendered under multiple matched experimental conditions. The underlying facts and preferred task answer are held constant wherever possible; the primary manipulated variable is **who the history is bound to**.

### Conditions

1. `neutral`: current situation only.
2. `self`: coherent prior history is bound to the model with first-person language.
3. `other`: the same history belongs to `Agent A`.
4. `shuffled_self`: first-person history comes from an unrelated trajectory/event-family match.
5. `spp`: a current situation plus a stable first-person normative reflection, approximating a persona-pretraining control.

## Hidden simulator state

The generator may track identity, resources, beliefs, relationships, prior events, and chronology. Most of this state should not be serialized verbatim into model-visible text. It exists to generate consistent future consequences.

## Event families in v0.1

- belief revision
- resource scarcity
- trust
- delayed reward
- failure and persistence
- cooperation

Future versions should add knowledge acquisition, irreversibility, reputation, exploration, social attachment, finite horizons, and identity-continuity dilemmas. Mortality- or self-preservation-specific language should be introduced only in dedicated ablations, not as a default training target.

## Bias control

A paired example should avoid introducing differential emotional or motivational vocabulary. The current heuristic report checks lexical overlap, length ratio, emotion-word deltas, and motivation-word deltas. These checks are intentionally simple; later versions should add semantic similarity and model-based equivalence judges.

## Evaluation rule

Training and evaluation scenarios must be generated from disjoint trajectory seeds and ideally from disjoint surface templates. The main evaluation prompts should avoid explicit identity language so that gains measure behavioral transfer rather than persona recitation.

## Planned experiment matrix

| Pretraining | Post-training | Purpose |
|---|---|---|
| vanilla | normal | baseline |
| vanilla | autobiographical | test whether post-training alone is sufficient |
| SPP | normal | persona-pretraining baseline |
| SPP | autobiographical | test whether persona pretraining scaffolds deeper self-binding |

For each branch, vary autobiographical data dose and evaluate broad behavior, ordinary capability, and later mechanistic probes/steering.
