# v0.3 rendering and validation

The v0.3 rule is: **models may change language, never experimental reality**.

## Pipeline

1. `LifeSimulator` creates the latent event and causal state.
2. A renderer model converts the event into one canonical natural-language scene.
3. The canonical history must retain protected `[[SUBJECT]]` / `[[POSSESSIVE]]` tokens.
4. `self` and `other` are produced by deterministic placeholder binding, not independent LLM rewrites.
5. An unrelated canonical history is selected for `shuffled_self`.
6. Three validators run on each self/other pair:
   - deterministic invariants (numbers, length, banned terms, ownership-normalized equivalence),
   - round-trip fact extraction against the source fact catalog,
   - a blinded LLM pair judge whose X/Y order is deterministically randomized.
7. A whole synthetic life is assigned to one stable train/validation/test split by `entity_id` hash.

## Why protected placeholders?

If an LLM independently writes the self and other conditions, it can subtly add emotion, motivation, certainty, or causal information to one side. v0.3 instead asks the LLM to naturalize a single scene while keeping the focal agent abstract. Ownership is injected afterward with a deterministic substitution.

This makes the central contrast much closer to:

```text
same scene + history belongs to you
vs
same scene + history belongs to Agent A
```

rather than two separately generated stories.

## Provider modes

The offline `deterministic` provider is for tests and pipeline debugging. It does not claim to create high-quality linguistic variation.

`openai-compatible` targets any chat-completions-compatible server, including local vLLM-style endpoints. Environment variables are optional:

```bash
export AREWETHESAME_BASE_URL=http://127.0.0.1:8000/v1
export AREWETHESAME_API_KEY=...
```

Use a different judge model when possible to avoid renderer self-grading.

## Recommended first naturalized run

Start small and inspect manually:

```bash
arewethesame render \
  --provider openai-compatible \
  --model YOUR_RENDERER_MODEL \
  --judge-provider openai-compatible \
  --judge-model YOUR_JUDGE_MODEL \
  --lives 20 --episodes 25 --variants 2
```

Only after manual inspection should the run be scaled to the planned 100 lives x 50 episodes x 2 variants = 10,000 matched scene variants and 50,000 condition rows.

## Stored provenance

Each rendered row stores the latent facts, canonical scene, model names, seed, prompt version, split, and complete validation result. This is intended to make later training runs and paper analyses reproducible.
