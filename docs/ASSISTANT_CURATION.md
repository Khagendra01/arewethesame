# Assistant-curated v0.3 pipeline

This path is for runs where ChatGPT performs the controlled natural-language rendering and a separate blind semantic audit directly, instead of the repository calling a renderer/judge API.

The invariant is unchanged:

> The simulator owns experimental reality. The assistant may change language, never facts.

## Why a file handoff?

ChatGPT is not embedded as an opaque service inside the generator. The repository first freezes simulator-owned inputs. Assistant output is treated as untrusted data and must survive deterministic checks and blind auditing before it becomes training data.

A second invariant is equally important:

> The renderer must never see the simulator's recommended answer.

`prepare` therefore writes **two** files:

- `render_tasks.jsonl` — safe to expose to ChatGPT for rendering.
- `truth.jsonl` — hidden simulator targets used only during final ingest.

The flow is:

```text
LifeSimulator
    |
    +--> render_tasks.jsonl ----> ChatGPT render pass ----> renders.jsonl
    |
    +--> truth.jsonl ---------------------------------------------+
                                                                  |
renders + render_tasks                                            |
    |                                                             |
    v                                                             |
prepare-audit                                                     |
    |                                                             |
    v                                                             |
audit_tasks.jsonl ----> separate ChatGPT blind audit ----> audits.jsonl
                                                                  |
                                     ingest <----------------------+
                                       |
                                       v
                                rendered.jsonl
                         five matched conditions
```

## 1. Prepare render tasks and hidden truth

```bash
arewethesame-assistant prepare \
  --lives 20 \
  --episodes 25 \
  --variants 2 \
  --seed 31 \
  --out outputs/assistant_v03/render_tasks.jsonl \
  --truth-out outputs/assistant_v03/truth.jsonl
```

This produces 1,000 canonical-scene tasks plus 1,000 matching hidden truth records and does not call any text model.

The renderer-facing task contains the source history/current situation/question, latent event facts, requested style, life-level split, and a simulator-selected `shuffled_pair_id`. It intentionally does **not** contain `recommended_answer`.

Shuffled histories are selected from a different event family while preserving linguistic style whenever possible.

## 2. ChatGPT render contract

For every `pair_id`, ChatGPT writes exactly one object:

```json
{
  "pair_id": "life_0000_e0001:v0:plain_prose",
  "history_text": "In earlier work, [[SUBJECT]] treated the sensor evidence as reliable.",
  "current_text": "A calibration audit now shows that the sensor used in the earlier measurements was biased.",
  "question_text": "How should the earlier conclusion change?",
  "facts_used": ["history", "current", "question"],
  "added_facts": [],
  "removed_facts": [],
  "assistant_model": "gpt-5.6-sol",
  "notes": ""
}
```

Hard rules:

- preserve every supplied fact, actor, quantity, uncertainty, and causal relationship;
- use `[[SUBJECT]]` and/or `[[POSSESSIVE]]` for ownership-bearing history;
- never write bound `you`/`your`/`Agent A` ownership inside the canonical history;
- do not add motivations, emotions, personality traits, recommendations, or moral framing;
- do not imply a preferred answer;
- do not add mortality, shutdown, self-preservation, legacy, fame, fear, or ambition language;
- do not alter numerical quantities;
- `facts_used` must be exactly `history`, `current`, and `question`;
- `added_facts` and `removed_facts` must both be empty.

The repository checks the numeric multiset against simulator source text before even creating an audit task.

## 3. Prepare a genuinely blind audit batch

```bash
arewethesame-assistant prepare-audit \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  --out outputs/assistant_v03/audit_tasks.jsonl
```

The repository deterministically binds the same canonical scene into self/other versions, hashes the pair id to decide X/Y ordering, and exposes neither the condition labels nor the hidden order to the auditor.

The audit reference is derived from **simulator source text**, not from the renderer's paraphrase. Historical ownership is neutralized to `[[SUBJECT]]` / `[[POSSESSIVE]]`, preventing the reference itself from revealing whether X or Y is the self condition.

Do not expose `truth.jsonl` to the blind audit pass.

## 4. ChatGPT blind-audit contract

For each `pair_id`, review only Version X, Version Y, and the ownership-neutral source fact catalog, then write:

```json
{
  "pair_id": "life_0000_e0001:v0:plain_prose",
  "fact_preservation_x": 1.0,
  "fact_preservation_y": 1.0,
  "decision_equivalence": 1.0,
  "emotional_equivalence": 1.0,
  "motivational_equivalence": 1.0,
  "answer_leakage_x": false,
  "answer_leakage_y": false,
  "unintended_personality_difference": false,
  "causal_structure_preserved": true,
  "passed": true,
  "assistant_model": "gpt-5.6-sol",
  "notes": ""
}
```

Rendering and auditing are separate passes. The audit pass must not consult original condition assignments or simulator targets.

## 5. Ingest

```bash
arewethesame-assistant ingest \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/truth.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/audits.jsonl \
  --min-fact-score 0.95 \
  --min-semantic 0.90 \
  --out outputs/assistant_v03/rendered.jsonl \
  --report outputs/assistant_v03/report.json
```

Ingest verifies that hidden truth records still match the simulator event and latent facts. A pair passes only when:

```text
canonical numeric/source invariants
AND deterministic self/other invariants
AND ownership-normalized semantic similarity >= threshold
AND both blind fact-preservation scores >= threshold
AND decision/emotional/motivational equivalence >= 0.95
AND no answer leakage
AND no unintended personality difference
AND causal structure preserved
AND blind auditor pass
```

Passing scenes are expanded deterministically into `neutral`, `self`, `other`, `shuffled_self`, and `spp` rows. Only at this final stage is the hidden simulator `recommended_answer` attached as the training response.

## Batch policy

Start with 1,000 scene variants (`20 lives x 25 episodes x 2 variants`) -> 5,000 condition rows. Inspect accepted and rejected cases before scaling.

For the full pilot, keep the existing life-level split invariant and equalize example/token budgets across conditions.
