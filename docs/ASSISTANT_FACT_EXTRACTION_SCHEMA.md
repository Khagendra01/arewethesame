# Assistant round-trip fact extraction contract

This contract is used only by the no-API ChatGPT-curated v0.3 path.

Input per pair:

- `pair_id`
- blind `version_x`
- blind `version_y`
- ownership-neutral simulator `fact_catalog`

The extractor must not receive condition labels, X/Y ownership order, simulator targets, `recommended_answer`, or blind-judge output.

Output per pair:

```json
{
  "pair_id": "...",
  "supported_fact_ids_x": ["history", "current", "question"],
  "supported_fact_ids_y": ["history", "current", "question"],
  "missing_fact_ids_x": [],
  "missing_fact_ids_y": [],
  "contradictions_x": [],
  "contradictions_y": [],
  "extracted_facts_x": {
    "history": {},
    "current": {},
    "question": {}
  },
  "extracted_facts_y": {
    "history": {},
    "current": {},
    "question": {}
  },
  "assistant_model": "gpt-5.6-sol",
  "notes": ""
}
```

Rules:

1. Extract what each version actually states; do not repair or reinterpret a missing fact.
2. `supported_fact_ids_*` may contain only IDs present in `fact_catalog`.
3. Every source fact ID not supported must appear in `missing_fact_ids_*`.
4. Every supported ID must have a corresponding entry in `extracted_facts_*`.
5. Record any factual or causal conflict with the source in `contradictions_*`.
6. Do not infer which version is self or other.
7. Do not consult the original render task, hidden truth, recommended answer, or pair-judge output during this pass.

The repository derives preservation scores from the supported/missing IDs. Under the default `0.95` threshold with three required fact groups, all three groups must survive on both X and Y. Any contradiction fails that side.

Only pairs passing this round-trip extraction stage are emitted to the blind pair judge.
