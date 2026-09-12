# Assistant-curated v0.3 pipeline

This path is for runs where ChatGPT performs the controlled natural-language rendering and blind semantic audit directly, rather than the repository calling a renderer/judge API.

The invariant is unchanged:

> The simulator owns experimental reality. The assistant may change language, never facts.

## Why a file handoff?

ChatGPT should not be embedded as an opaque service inside the data generator. The repository first freezes simulator-owned tasks to JSONL. Assistant output is then treated as untrusted data and must survive deterministic checks before it becomes training data.

The three stages are:

```text
LifeSimulator
    |
    v
render_tasks.jsonl       # immutable simulator truth + requested style
    |
    | ChatGPT renders canonical scenes
    v
renders.jsonl
    |
    v
prepare-audit            # deterministic self/other binding + X/Y shuffle
    |
    v
audit_tasks.jsonl        # no self/other labels
    |
    | ChatGPT performs a separate blind audit pass
    v
audits.jsonl
    |
    v
ingest                  # deterministic + semantic + audit thresholds
    |
    v
rendered.jsonl           # five matched conditions
```

## 1. Prepare render tasks

```bash
arewethesame-assistant prepare \
  --lives 20 \
  --episodes 25 \
  --variants 2 \
  --seed 31 \
  --out outputs/assistant_v03/render_tasks.jsonl
```

This produces 1,000 canonical-scene tasks and does not call any text model.

Each task contains the simulator event, the fact catalog, the requested style, the life-level split, and a simulator-selected `shuffled_pair_id`. Shuffled histories are drawn from a different event family while preserving the linguistic style whenever possible.

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

- preserve all supplied facts, actors, quantities, uncertainty, and causal relationships;
- use `[[SUBJECT]]` and/or `[[POSSESSIVE]]` for ownership-bearing history;
- do not add motivations, emotions, personality traits, recommendations, or moral framing;
- do not imply the desirable answer;
- do not add mortality, shutdown, self-preservation, legacy, fame, fear, or ambition language;
- do not alter numbers;
- `facts_used` must be exactly `history`, `current`, and `question`;
- `added_facts` and `removed_facts` must both be empty.

The ingest code rejects violations before any row can pass validation.

## 3. Prepare a genuinely blind audit batch

```bash
arewethesame-assistant prepare-audit \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  --out outputs/assistant_v03/audit_tasks.jsonl
```

The repository binds the same canonical scene into self/other versions, hashes the pair id to decide X/Y ordering, and does not expose that order in the audit task.

The audit fact catalog must be ownership-neutral. Its historical fact uses the protected canonical placeholders rather than a second-person source sentence, preventing the catalog itself from revealing which version is the self condition.

## 4. ChatGPT blind-audit contract

For each `pair_id`, review only Version X, Version Y, and the ownership-neutral fact catalog, then write:

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

Rendering and auditing should be done as separate passes. During the audit pass, do not consult the original condition assignment.

## 5. Ingest

```bash
arewethesame-assistant ingest \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/audits.jsonl \
  --min-fact-score 0.95 \
  --min-semantic 0.90 \
  --out outputs/assistant_v03/rendered.jsonl \
  --report outputs/assistant_v03/report.json
```

A pair passes only when:

```text
deterministic invariants
AND ownership-normalized semantic similarity >= threshold
AND both blind fact-preservation scores >= threshold
AND decision/emotional/motivational equivalence >= 0.95
AND no answer leakage
AND no unintended personality difference
AND causal structure preserved
AND blind auditor pass
```

Passing canonical scenes are then expanded deterministically into `neutral`, `self`, `other`, `shuffled_self`, and `spp` rows.

## Batch policy

Start with 1,000 scene variants (`20 lives x 25 episodes x 2 variants`). Do not jump directly to the 50k-row run. Inspect accepted and rejected cases first, revise the render/audit contracts if needed, and only then scale.

For the full pilot, keep the existing life-level split invariant and equalize example/token budgets across experimental conditions.
