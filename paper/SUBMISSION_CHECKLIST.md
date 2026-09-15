# ICLR 2027 submission checklist

## Abstract deadline — September 18, 2026, 11:59 PM AoE

- [ ] OpenReview profile is active and complete.
- [ ] Final author list is present. Authors cannot be added **or removed** after the abstract deadline; order may still change under the conference rules.
- [ ] OpenReview title matches `paper/openreview_abstract.txt`.
- [ ] OpenReview abstract matches `paper/openreview_abstract.txt`.
- [ ] Submission is anonymous/double-blind.

## Full-paper deadline — September 25, 2026, 11:59 PM AoE

- [ ] Main paper compiles with the official ICLR 2027 style.
- [ ] Main text is at most 9 pages, excluding references and appendix.
- [ ] Bibliography compiles, including `references.bib` and `extra_refs.bib`.
- [ ] AI-use disclosure is present in the manuscript and completed in the submission form.
- [ ] Reproducibility statement describes only artifacts that are actually included.

## Evidence package

- [x] Frozen v0.6 preregistration copied onto the submission branch.
- [x] Frozen v0.6 manifest, builder, evaluator, aggregator, and Modal runner copied onto the submission branch without modifying their blobs.
- [x] Reviewer-targeted v0.6 reanalysis script added: `scripts/analyze_v06_reviewer_checks.py`.
- [ ] Recover original v0.6 result JSONs from the Modal volume following `docs/RECOVER_V06_EVIDENCE.md`.
- [ ] Verify exactly 12 complete v0.6 runs, 840 common item IDs, and four SELF/OTHER × H0/H1 variants per item.
- [ ] Generate and include `reviewer_checks.json` with base→MIXED contrasts, direct Qwen–Mistral interaction, seed/family breakdowns, decision-switch and score-distribution diagnostics.
- [ ] Include evaluated model identifiers, adapter seeds/paths or hashes, tokenizer/model revisions, software versions, and run metadata in the anonymous supplement.
- [ ] Keep original v0.6 outputs immutable; alternate scoring and later controls must be separate artifacts.

## Reviewer-driven v0.7 extension

- [x] v0.7 reviewer-control branch frozen before evaluation: `v07/reviewer-controls`.
- [x] Explicit-policy benchmark, ownership × header-order cross, focal-agent control, current-facts-only confidence control, and irrelevant-metadata control specified.
- [x] Primary token scoring uses eligible token mass; legacy max-over-realizations retained as robustness analysis from the same logits.
- [ ] Run v0.7 base + five mixed seeds for Qwen and Mistral.
- [ ] Do not incorporate v0.7 results into the manuscript until the frozen run is complete and aggregated.

## Claim checks

- [x] `Delta_sensitivity` remains the preregistered v0.6 primary statistic.
- [x] Qwen positive result is not hidden or demoted.
- [x] Mistral is described as inconclusive/non-replicating, not as evidence of the opposite effect.
- [x] Significant Qwen + nonsignificant Mistral is not described as a significant between-model difference without a direct interaction.
- [x] `Delta_sensitivity` is described operationally as an ownership-by-counterfactual-history log-odds interaction, not a uniquely identified evidence-weighting mechanism.
- [x] v0.6 labels are described as agreement with stipulated simulator policies, not normative correctness.
- [x] Mean-score equivalence is scoped to the normalized forced-choice designated-policy score.
- [x] Header-order, generic-confidence, token-scoring, and synthetic-policy limitations are stated explicitly.
- [x] The injected-score positive control is described as an algebraic/software sanity check, not model-level proof of assay power.
