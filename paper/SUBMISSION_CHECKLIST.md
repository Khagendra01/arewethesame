# ICLR 2027 submission checklist

## Frozen conference requirements

- Abstract deadline: **September 18, 2026, 11:59 PM AoE**.
- Full-paper deadline: **September 25, 2026, 11:59 PM AoE**.
- No authors may be added after the abstract deadline. Author order may still change before the full-paper deadline.
- Submission is double blind; the manuscript and supplementary text must remain anonymous.
- Main text limit: **9 pages**. References do not count. Appendices may follow the references.
- All authors need current OpenReview profiles.
- ICLR 2027 requires an AI-use disclosure both in the manuscript and in the OpenReview submission form.

## Repository submission state

- [x] ICLR 2027 LaTeX template in `paper/`.
- [x] Anonymous author block retained for double blind.
- [x] v0.6 title and abstract synchronized between `paper/main.tex` and `paper/openreview_abstract.txt`.
- [x] v0.5 chronology preserved: original ID test preregistered; multi-seed/OOD/Mistral extensions labeled as subsequent robustness analyses.
- [x] v0.6 chronology preserved: identity-controlled benchmark and interpretation rules frozen before v0.6 evaluation.
- [x] v0.6 primary statistic remains `Delta_sensitivity`; the positive Qwen result is not post-hoc demoted.
- [x] Probability-scale equivalence claims are restricted to the prospectively specified v0.6 `Delta_prob` analysis.
- [x] Ceiling objection addressed with non-saturated v0.6 accuracy (~0.65).
- [x] `you` vs `Agent A` surface-form objection addressed with byte-identical history bodies and assigned aliases.
- [x] Crossed hierarchical seed × item bootstrap described explicitly.
- [x] Injected-logit positive controls described as assay-sensitivity checks, not learnability evidence.
- [x] Closest self-preference / self-recognition / assigned-identity literature added.
- [x] SPP baseline renamed/described as an **SPP-inspired reflection control**, not a reproduction of Synthetic Persona Pretraining.
- [x] Ethics / broader-impact section included.
- [x] Required AI-use statement included.
- [x] Reproducibility statement updated for v0.6.
- [x] Local LaTeX syntax/layout check passes; appendix starts on page 8 in the compile check, leaving the counted main text below the 9-page limit.

## Human/account actions still required

- [ ] Confirm the complete author list before the abstract deadline.
- [ ] Confirm every author has a current OpenReview profile.
- [ ] Paste the exact title/abstract from `paper/openreview_abstract.txt` into the ICLR 2027 OpenReview submission.
- [ ] Complete the OpenReview AI-use disclosure consistently with the manuscript statement.
- [ ] Upload the final anonymous PDF and any supplementary code/material by the full-paper deadline.
- [ ] Recompile on Overleaf or a complete local TeX installation with BibTeX before upload and check for undefined citations/references.
