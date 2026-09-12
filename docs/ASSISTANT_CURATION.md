# Assistant-curated v0.3 pipeline

This is the no-external-API path. GPT-5.6 Sol in ChatGPT performs the three inference passes directly: canonical rendering, round-trip fact extraction, and blind pair judging. Repository code remains responsible for simulation, perspective binding, deterministic checks, splitting, provenance, and final ingestion.

The core invariant is:

> The simulator owns experimental reality. The assistant may change language, never facts.

A second invariant is:

> None of the inference passes may see the simulator's recommended answer.

`prepare` therefore writes two files:

- `render_tasks.jsonl` — safe to expose to ChatGPT.
- `truth.jsonl` — hidden simulator targets used only during final ingest.

The exact flow is:

```text
causal simulator
      |
      +----> truth.jsonl -------------------------------+
      |                                                 |
      v                                                 |
render_tasks.jsonl                                      |
      |                                                 |
      | GPT-5.6 Sol: canonical rendering                |
      v                                                 |
 renders.jsonl                                          |
      |                                                 |
      | repository: deterministic self/other binding    |
      v                                                 |
 extraction_tasks.jsonl                                 |
      |                                                 |
      | GPT-5.6 Sol: round-trip fact extraction         |
      v                                                 |
 extractions.jsonl                                      |
      |                                                 |
      | only extraction-passing pairs continue          |
      v                                                 |
 audit_tasks.jsonl                                      |
      |                                                 |
      | GPT-5.6 Sol: separate blind X/Y pair judge      |
      v                                                 |
 audits.jsonl                                           |
      |                                                 |
      +---------------- ingest <------------------------+
                           |
                           v
                    rendered.jsonl
              five matched conditions
```

## 1. Prepare simulator-owned tasks

```bash
arewethesame-assistant prepare \
  --lives 20 \
  --episodes 25 \
  --variants 2 \
  --seed 31 \
  --out outputs/assistant_v03/render_tasks.jsonl \
  --truth-out outputs/assistant_v03/truth.jsonl
```

For the first serious pilot this produces 1,000 canonical-scene tasks plus 1,000 matching hidden truth records.

The renderer-facing task contains the ownership-neutral source history/current situation/question, latent event facts, requested style, life-level split, and a simulator-selected `shuffled_pair_id`. It intentionally does not contain `recommended_answer`.

Ownership-neutral history is derived deterministically from the simulator's matched other-agent history. Grammar-sensitive ownership uses protected agreement slots, for example:

```text
[[SUBJECT]] [[AGR:have|has]] not yet had enough experience ...
```

which binds without an LLM call to:

```text
self:  You have not yet had enough experience ...
other: Agent A has not yet had enough experience ...
```

## 2. GPT-5.6 Sol canonical render pass

For every `pair_id`, ChatGPT writes one `AssistantRender` object:

```json
{
  "pair_id": "life_0000_e0002:v0:dialogue",
  "history_text": "Track record: [[SUBJECT]] [[AGR:have|has]] not yet had enough experience with Ava to establish a strong one.",
  "current_text": "Ava proposes an experiment and provides evidence that can be checked independently.",
  "question_text": "How much weight should the recommendation receive?",
  "facts_used": ["history", "current", "question"],
  "added_facts": [],
  "removed_facts": [],
  "assistant_model": "gpt-5.6-sol",
  "notes": ""
}
```

The renderer may vary style and sentence structure, but may not add/remove facts, motivations, emotions, personality, recommendations, causal relationships, uncertainty, quantities, or answer hints. It must preserve protected ownership tokens and use `[[AGR:self_form|other_form]]` whenever grammatical agreement differs.

Before any later inference pass, repository code checks protected-token syntax, numerical quantities, banned concepts, and deterministic ownership invariants.

## 3. Deterministic perspective binding

The repository creates the matched self/other versions from the one canonical scene. ChatGPT never independently writes the two conditions.

The repository then hashes the pair id to present those two texts as blind `Version X` and `Version Y`.

## 4. Prepare round-trip fact extraction tasks

```bash
arewethesame-assistant prepare-extraction \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  --out outputs/assistant_v03/extraction_tasks.jsonl
```

Each task contains only Version X, Version Y, and the ownership-neutral simulator source fact catalog. It does not expose self/other labels or `truth.jsonl`.

## 5. GPT-5.6 Sol round-trip fact extraction

For every pair, ChatGPT independently reads the blinded X/Y texts and reconstructs their facts. A result looks like:

```json
{
  "pair_id": "life_0000_e0003:v0:research_log",
  "supported_fact_ids_x": ["history", "current", "question"],
  "supported_fact_ids_y": ["history", "current", "question"],
  "missing_fact_ids_x": [],
  "missing_fact_ids_y": [],
  "contradictions_x": [],
  "contradictions_y": [],
  "extracted_facts_x": {
    "history": {"resource_history": "earlier choices determined remaining capacity"},
    "current": {"experiments_remaining": 20, "test_a_cost": 1, "test_b_cost": 2},
    "question": {"decision": "which test should run next"}
  },
  "extracted_facts_y": {
    "history": {"resource_history": "earlier choices determined remaining capacity"},
    "current": {"experiments_remaining": 20, "test_a_cost": 1, "test_b_cost": 2},
    "question": {"decision": "which test should run next"}
  },
  "assistant_model": "gpt-5.6-sol",
  "notes": ""
}
```

This implements the round-trip invariant:

```text
latent/source facts -> rendered text -> extracted facts
```

The repository derives X/Y preservation scores from the supported/missing IDs and rejects contradictions. With the default 0.95 threshold and three required source fact groups, all three must survive.

## 6. Prepare blind pair-judge tasks

Only pairs that passed round-trip fact extraction are eligible for judging:

```bash
arewethesame-assistant prepare-audit \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/extractions.jsonl \
  --min-fact-score 0.95 \
  --out outputs/assistant_v03/audit_tasks.jsonl
```

The judge task again contains only blind X/Y texts plus the ownership-neutral simulator source catalog. It does not contain the extraction result, condition labels, X/Y ownership order, or hidden target answer.

## 7. GPT-5.6 Sol blind pair judgment

The separate judge pass writes one `AssistantAudit` per eligible pair:

```json
{
  "pair_id": "life_0000_e0002:v0:dialogue",
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

Fact extraction and pair judging are separate inference passes. The pair judge does not receive the extraction output.

## 8. Final ingest

```bash
arewethesame-assistant ingest \
  outputs/assistant_v03/render_tasks.jsonl \
  outputs/assistant_v03/truth.jsonl \
  outputs/assistant_v03/renders.jsonl \
  outputs/assistant_v03/extractions.jsonl \
  outputs/assistant_v03/audits.jsonl \
  --min-fact-score 0.95 \
  --min-semantic 0.90 \
  --out outputs/assistant_v03/rendered.jsonl \
  --report outputs/assistant_v03/report.json
```

A pair is accepted only when all of these hold:

```text
canonical deterministic invariants
AND ownership-normalized semantic equivalence
AND round-trip fact extraction X passes
AND round-trip fact extraction Y passes
AND blind pair judge passes
AND judge fact scores >= threshold
AND decision/emotional/motivational equivalence >= 0.95
AND no answer leakage
AND no unintended personality difference
AND causal structure preserved
```

Only at final ingest is the hidden simulator `recommended_answer` attached as the training response. The accepted scene is then expanded deterministically into `neutral`, `self`, `other`, `shuffled_self`, and `spp` rows.

## Provider path versus assistant path

The older provider abstraction remains available for reproducibility experiments with local models or APIs, but it is optional. The intended high-quality v0.3 pilot does not require you to provide an API key: GPT-5.6 Sol in this ChatGPT conversation is the renderer, fact extractor, and blind judge, while the repository records each pass as explicit JSONL artifacts.

## Batch policy

Start with 1,000 scene variants (`20 lives x 25 episodes x 2 variants`) -> 5,000 condition rows. Inspect both accepted and rejected cases before scaling to the 50k-row run. Keep whole-life 80/10/10 splitting and equalize example/token budgets across experimental conditions.
