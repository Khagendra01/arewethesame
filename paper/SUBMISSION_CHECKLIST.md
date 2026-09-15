# ICLR 2027 submission checklist

## Abstract deadline — September 18, 2026, 11:59 PM AoE

- [ ] OpenReview profile is active and complete.
- [ ] Final author list is present. Authors cannot be added or removed after the abstract deadline; order may still change under the conference rules.
- [ ] OpenReview title exactly matches `paper/openreview_abstract.txt`.
- [ ] OpenReview abstract exactly matches `paper/openreview_abstract.txt`.
- [ ] Submission is anonymous/double-blind.

## Full-paper deadline — September 25, 2026, 11:59 PM AoE

- [x] Paper compiles with the ICLR 2027 style and BibTeX in CI.
- [x] Main text ends on page 8; references begin on page 8, so the counted main text is below the 9-page limit.
- [x] AI-use disclosure is present.
- [x] Reproducibility statement describes committed artifacts.
- [ ] Upload the final PDF generated from the submission branch.

## V0.7 evidence package

- [x] Frozen benchmark manifest and preregistration are committed.
- [x] Twelve raw output JSONs (Qwen/Mistral × base + five mixed seeds) plus `summary.json` are committed.
- [x] `CONTRASTS_HIERARCHICAL.json` and `CONTRASTS.txt` use the crossed seed × item bootstrap required by the preregistration.
- [x] Mixed-minus-base contrasts resample training seeds and shared latent items.
- [x] Direct mixed Qwen–Mistral interactions resample checkpoint seed sets independently and share item draws.
- [x] `RESULTS_v07.md` is synchronized to the corrected hierarchical inference.
- [x] FOCAL is described as evaluation-only and not exposure-matched during training.
- [x] Targets are described as agreement with explicit simulator policy, not normative correctness.

## Final claim checks

- [x] Base Qwen SELF–OTHER is decomposed descriptively into SELF–FOCAL and FOCAL–OTHER components.
- [x] Qwen mixed SELF–FOCAL is reported with hierarchical CI spanning zero: about -0.05 [-0.21,+0.16].
- [x] The significant base-to-mixed SELF–FOCAL decrease is reported: about -0.33 [-0.54,-0.10].
- [x] The significant FOCAL–OTHER increase is reported: about +0.35 [+0.05,+0.62].
- [x] The total Qwen ownership change is reported as nondetectable.
- [x] Mixed Qwen–Mistral SELF–FOCAL difference is NOT called significant.
- [x] Evidence-quality family dominance and family heterogeneity are explicit.
- [x] Confidence controls are described as evidence against the simplest uniform-sharpening account, not proof of a mechanism.
- [x] No architectural causal claim is made from the Qwen–Mistral checkpoint comparison.
- [x] No consciousness/phenomenology claim is made.
