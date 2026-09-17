# ICLR 2027 submission checklist

## Abstract deadline — September 18, 2026, 11:59 PM AoE

- [ ] OpenReview profile is active and complete.
- [ ] Final author list is present. Authors cannot be added or removed after the abstract deadline; order may still change under the conference rules.
- [ ] OpenReview title exactly matches `paper/openreview_abstract.txt`.
- [ ] OpenReview abstract exactly matches `paper/openreview_abstract.txt`.
- [ ] Submission is anonymous/double-blind.

## Full-paper deadline — September 25, 2026, 11:59 PM AoE

- [x] Paper compiles with the ICLR 2027 style and BibTeX in CI.
- [x] Counted manuscript content stays within the 9-page main-text limit: the conclusion and AI-use statement end on page 8, the reproducibility statement occupies the top of page 9, and references begin on page 9.
- [x] AI-use disclosure explicitly covers methodological critique/design feedback, implementation, orchestration, statistical-analysis code and interpretation, literature search, and drafting/editing.
- [x] Reproducibility statement describes the frozen supplementary materials.
- [x] Title line breaks are fixed so `Focal-Agent Effects` is not hyphenated across lines.
- [x] CI builds an anonymized supplementary ZIP and scans it for author/repository identifiers.
- [x] CI regenerates the frozen v0.7 benchmark and verifies SHA-256 `d4960832076326ba5ed31ff3664095dea8c916ea3fabf7e482bcde8207f5cc7a` before packaging.
- [ ] Upload the final PDF generated from the submission branch.
- [ ] Upload the anonymous supplementary ZIP rather than linking the identifying public development repository.

## V0.7 evidence package

- [x] Frozen benchmark manifest and preregistration are committed.
- [x] Twelve raw output JSONs (Qwen/Mistral × base + five mixed seeds) plus `summary.json` are committed.
- [x] `CONTRASTS_HIERARCHICAL.json` and `CONTRASTS.txt` use the crossed seed × item bootstrap required by the preregistration.
- [x] Mixed-minus-base contrasts resample training seeds and shared latent items.
- [x] Direct mixed Qwen–Mistral interactions resample checkpoint seed sets independently and share item draws.
- [x] `RESULTS_v07.md` is synchronized to the corrected hierarchical inference.
- [x] The manuscript discloses the preregistered 5,000 bootstrap draws and the final 10,000-draw analysis, with the resampling scheme and estimands unchanged.
- [x] The preregistered legacy max-token scoring robustness result is reported and preserves the qualitative Qwen mixed pattern.
- [x] Per-family and leave-one-family-out analyses are explicitly labeled exploratory.
- [x] FOCAL is described as evaluation-only and not exposure-matched during training.
- [x] Targets are described as agreement with explicit simulator policy, not normative correctness.

## Final claim checks

- [x] Base Qwen SELF–OTHER is decomposed descriptively into SELF–FOCAL and FOCAL–OTHER components.
- [x] Qwen mixed SELF–FOCAL is reported with hierarchical CI spanning zero: about -0.05 [-0.21,+0.16].
- [x] The significant base-to-mixed SELF–FOCAL decrease is reported: about -0.33 [-0.54,-0.10].
- [x] The significant FOCAL–OTHER increase is reported: about +0.35 [+0.05,+0.62].
- [x] The total Qwen ownership change is described as not detected, rather than as statistical equivalence.
- [x] Mixed Qwen–Mistral SELF–FOCAL difference is NOT called significant.
- [x] Evidence-quality family dominance and family heterogeneity are explicit.
- [x] Confidence controls are described as evidence against the simplest uniform-sharpening account, not proof of a mechanism.
- [x] No architectural causal claim is made from the Qwen–Mistral checkpoint comparison.
- [x] No consciousness/phenomenology claim is made.


## 2026-09-17 submission revision

- [x] Preserve frozen v0.7 items, raw outputs, primary estimates, and preregistration.
- [x] Define all four task families, option scoring, counterfactual statistic, current-facts margin, and irrelevant-metadata control precisely.
- [x] Add role/exposure caveat for FOCAL and shared-recipe caveat for cross-checkpoint comparison.
- [x] Add actual recoverable training/evaluation settings and disclose unavailable revision/version metadata.
- [x] Add post-hoc CPU-only absolute-performance, matched-switch, and leave-one-training-seed-out diagnostics.
- [x] Remove 95%-CI triple-star notation and label scoring conventions in result tables.
- [ ] Final artifact hashes and extracted-ZIP verification: recorded in the CI-generated revision-and-verification report for the final commit.
- [ ] OpenReview PDF/supplement upload and author attestations: author action only.
