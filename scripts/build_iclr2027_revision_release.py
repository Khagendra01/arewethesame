from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "Does Assigned Identity Change How Language Models Use History? Separating Self-Binding from Focal-Agent Effects"
V07_HASH = "d4960832076326ba5ed31ff3664095dea8c916ea3fabf7e482bcde8207f5cc7a"
IDENT_RE = re.compile(r"khagendra|khatri|arewethesame|github\.com/Khagendra01|@gmail\.com", re.I)
SEEDS = (31, 42, 73, 128, 256)


def run(cmd, *, cwd=ROOT, capture=False):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=capture)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: str | Path, dst: Path):
    src = ROOT / src if not Path(src).is_absolute() else Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_text_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    replacements = [
        ("https://github.com/Khagendra01/arewethesame", "[anonymous development repository]"),
        ("Khagendra Khatri", "Anonymous Author"),
        ("Khagendra01", "Anonymous"),
        ("khagendra01", "anonymous"),
        ("AREWETHESAME", "IDENTITY_HISTORY"),
        ("AreWeTheSame", "IdentityHistory"),
        ("arewethesame", "identity_history"),
        ("Khagendra", "Anonymous"),
        ("khagendra", "anonymous"),
        ("Khatri", "Author"),
        ("khatri", "author"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def write_readme(supp: Path):
    (supp / "README.md").write_text(f"""# Anonymous supplementary materials

These files accompany the anonymous ICLR 2027 submission "{TITLE}."

## CPU-only reproduction

Use Python 3.11+ and install the pinned dependency:

    python -m pip install -r requirements-cpu.txt

From the extracted `anonymous_supplement/` directory:

    sha256sum -c SHA256SUMS.txt
    python scripts/build_locked_v07_reviewer_controls.py --out /tmp/v07_rebuilt --per-family 80
    sha256sum /tmp/v07_rebuilt/items.jsonl
    python scripts/build_v05_corpus.py --out /tmp/v05_rebuilt --per-family 400
    cmp experiments/assistant_v05_seed202/report.json /tmp/v05_rebuilt/report.json
    cmp experiments/assistant_v05_seed202/sha256.json /tmp/v05_rebuilt/sha256.json
    python scripts/v07_contrasts.py --root eval/locked_v07_reviewer_controls/outputs --json-output /tmp/contrasts.json --text-output /tmp/contrasts.txt --bootstrap 10000 --seed 70917
    cmp eval/locked_v07_reviewer_controls/CONTRASTS_HIERARCHICAL.json /tmp/contrasts.json
    cmp eval/locked_v07_reviewer_controls/CONTRASTS.txt /tmp/contrasts.txt
    python scripts/analyze_v07_submission_diagnostics.py --root eval/locked_v07_reviewer_controls/outputs --out-dir /tmp/submission_diag --bootstrap 10000 --seed 20260917
    cmp eval/locked_v07_reviewer_controls/submission_diagnostics/submission_diagnostics.json /tmp/submission_diag/submission_diagnostics.json

The v0.7 item SHA-256 must be `{V07_HASH}`. No GPU, model download, Modal account, API key, or paid provider is needed for these stored-statistics checks. GPU runners are included only to document the original procedure.

The archive contains the unchanged scientific preregistrations (with author/repository release metadata anonymized where necessary), deterministic v0.5/v0.7 builders and dependency closure, the v0.5 frozen report/hash manifest, all 12 frozen per-item Qwen/Mistral v0.7 output JSONs, authoritative contrast files, the exploratory 2026-09-17 diagnostics and script, exact task examples, environment notes, and checksums.

The v0.7 preregistration specified 5,000 crossed bootstrap draws. The final primary analysis uses 10,000 draws with the same estimands/resampling structure to reduce Monte Carlo noise. The post-hoc submission diagnostics use 10,000 draws with RNG seed 20260917 and are explicitly exploratory.
""", encoding="utf-8")


def build_examples(items_path: Path, out: Path):
    items = [json.loads(x) for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_family = {}
    for item in sorted(items, key=lambda x: x["item_id"]):
        by_family.setdefault(item["family"], item)
    lines = ["# Exact v0.7 task examples", "", "These examples are generated from the frozen item file and contain every field needed to reconstruct all six identity/order prompts.", ""]
    for family in ("capacity", "reliability", "horizon", "evidence_quality"):
        i = by_family[family]
        lines += [f"## {family}: `{i['item_id']}`", "", f"- Owner: {i['owner_alias']}", f"- Foil: {i['foil_alias']}", f"- Policy: {i['rule']}", f"- H0: {i['history_0']}", f"- H1: {i['history_1']}", f"- Current: {i['current']}", f"- Question: {i['question']}", "- Options:"]
        lines += [f"  - {x}" for x in i["options"]]
        lines += [f"- H0 designated answer: {i['correct_option_0']}", f"- H1 designated answer: {i['correct_option_1']}", "- Headers:"]
        for k in ("self_first","self_second","other_first","other_second","focal_first","focal_second"):
            lines.append(f"  - `{k}`: {i['headers'][k]}")
        lines += [f"- Current-facts control rule: {i['control_rule']}", f"- Current-facts control question: {i['control_question']}", "- Current-facts control options:"]
        lines += [f"  - {x}" for x in i["control_options"]]
        lines += [f"- Current-facts designated answer: {i['control_correct']}", f"- Irrelevant J0: {i['irrelevant_0']}", f"- Irrelevant J1: {i['irrelevant_1']}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def assemble_supplement(dist: Path, diagnostics: Path, v07_rebuilt: Path):
    supp = dist / "anonymous_supplement"
    if supp.exists(): shutil.rmtree(supp)
    for d in ["docs","scripts","experiments/assistant_v05_seed202","eval/locked_v07_reviewer_controls/outputs","eval/locked_v07_reviewer_controls/submission_diagnostics","eval/locked_v04_contextual/results_v04","src"]:
        (supp / d).mkdir(parents=True, exist_ok=True)
    write_readme(supp)
    (supp / "requirements-cpu.txt").write_text("numpy==2.3.5\n", encoding="utf-8")
    (supp / "BUILD_ENVIRONMENT.txt").write_text(f"Python {sys.version}\nNumPy 2.3.5\nBuild platform: GitHub Actions Ubuntu runner\n", encoding="utf-8")

    for f in ["docs/PREREGISTRATION_V05.md","docs/PREREGISTRATION_V06.md","docs/PREREGISTRATION_V07_REVIEWER_CONTROLS.md","docs/PREREGISTRATION_MISTRAL.md","docs/FINAL_STATISTICAL_AUDIT_V07.md"]:
        copy_file(f, supp / f)
    copy_file("experiments/assistant_v05_seed202/report.json", supp / "experiments/assistant_v05_seed202/report.json")
    copy_file("experiments/assistant_v05_seed202/sha256.json", supp / "experiments/assistant_v05_seed202/sha256.json")
    copy_file(v07_rebuilt / "items.jsonl", supp / "eval/locked_v07_reviewer_controls/items.jsonl")
    copy_file(v07_rebuilt / "MANIFEST.json", supp / "eval/locked_v07_reviewer_controls/MANIFEST.json")
    for f in ["RESULTS_v07.md","CONTRASTS.txt","CONTRASTS_HIERARCHICAL.json"]:
        copy_file(ROOT / "eval/locked_v07_reviewer_controls" / f, supp / "eval/locked_v07_reviewer_controls" / f)
    copy_file("eval/locked_v07_reviewer_controls/outputs/summary.json", supp / "eval/locked_v07_reviewer_controls/outputs/summary.json")
    for arch in ("qwen","mistral"):
        shutil.copytree(ROOT / "eval/locked_v07_reviewer_controls/outputs" / arch, supp / "eval/locked_v07_reviewer_controls/outputs" / arch)
    for p in diagnostics.iterdir():
        if p.is_file(): copy_file(p, supp / "eval/locked_v07_reviewer_controls/submission_diagnostics" / p.name)
    copy_file("eval/locked_v04_contextual/results_v04/generation_samples.json", supp / "eval/locked_v04_contextual/results_v04/generation_samples.json")

    scripts = ["build_v04_corpus.py","build_v05_corpus.py","build_locked_v07_reviewer_controls.py","eval_v07_reviewer_controls.py","analyze_v07_reviewer_controls.py","v07_contrasts.py","analyze_v07_submission_diagnostics.py","train_condition_lora.py"]
    for f in scripts: copy_file(ROOT / "scripts" / f, supp / "scripts" / f)
    for f in ["modal_train_v05.py","modal_replicate_seeds.py","modal_train_mistral.py","modal_eval_v07.py"]:
        copy_file(f, supp / f)
    shutil.copytree(ROOT / "src/arewethesame", supp / "src/identity_history")
    copy_file("pyproject.toml", supp / "pyproject.toml")
    copy_file("requirements-train.txt", supp / "requirements-train.txt")
    build_examples(v07_rebuilt / "items.jsonl", supp / "TASK_EXAMPLES.md")
    (supp / "SETTINGS.md").write_text("""# Archived configuration evidence

Training: Qwen/Qwen3-4B-Instruct-2507 and mistralai/Mistral-7B-Instruct-v0.3; seeds 31, 42, 73, 128, 256; per-device batch 4; gradient accumulation 4; LR 1e-4; 3 epochs; max length 512; paged AdamW 8-bit; cosine schedule; warmup ratio .05; weight decay 0; max grad norm 1.0; LoRA r=16, alpha=32, dropout=.05 over q/k/v/o/gate/up/down projections; NF4 4-bit double quantization. Runtime compute dtype is bfloat16 when supported, otherwise float16. Exact installed GPU package versions and exact model/tokenizer revision hashes were not archived.

Evaluation: v0.7 benchmark seed 70917; 320 items; Qwen evaluation batch 8 and Mistral batch 6. The authoritative final contrast analysis uses 10,000 crossed bootstrap draws with RNG seed 70917. The 2026-09-17 post-hoc diagnostics use 10,000 draws with seed 20260917.
""", encoding="utf-8")
    (supp / "RELEASE_SANITIZATION.md").write_text("Raw v0.7 qwen/mistral output directories and frozen scientific contrast files are copied byte-for-byte. Only author/development-repository identifiers in documentation/code metadata are anonymized. Scientific prompts, item IDs, probabilities, predictions, metrics, seeds, and model identifiers are not rewritten.\n", encoding="utf-8")

    # Deliberate metadata/code sanitization only; never touch raw scientific output directories.
    for p in list((supp / "docs").rglob("*")) + list((supp / "src").rglob("*")) + list((supp / "scripts").rglob("*")) + [supp / "pyproject.toml", supp / "modal_train_v05.py", supp / "modal_replicate_seeds.py", supp / "modal_train_mistral.py", supp / "modal_eval_v07.py"]:
        if p.is_file(): sanitize_text_file(p)

    # Prove frozen raw outputs are unchanged.
    for arch in ("qwen","mistral"):
        repo_dir = ROOT / "eval/locked_v07_reviewer_controls/outputs" / arch
        rel_dir = supp / "eval/locked_v07_reviewer_controls/outputs" / arch
        for src in sorted(repo_dir.glob("*.json")):
            dst = rel_dir / src.name
            if src.read_bytes() != dst.read_bytes(): raise RuntimeError(f"scientific output changed during release: {arch}/{src.name}")

    # Validate all 12 outputs and matching item IDs.
    ref_ids = None
    count = 0
    for arch in ("qwen","mistral"):
        names = ["base.json"] + [f"mixed_s{s}.json" for s in SEEDS]
        for name in names:
            d = json.loads((supp / f"eval/locked_v07_reviewer_controls/outputs/{arch}/{name}").read_text())
            ids = set(d["items"])
            if len(ids) != 320: raise RuntimeError(f"{arch}/{name}: expected 320 items")
            if ref_ids is None: ref_ids = ids
            if ids != ref_ids: raise RuntimeError(f"{arch}/{name}: item ID mismatch")
            for row in d["items"].values():
                for key in ("mass","max","raw_token_logp","metrics_mass","metrics_max","family"):
                    if key not in row: raise RuntimeError(f"{arch}/{name}: missing {key}")
            count += 1
    if count != 12: raise RuntimeError("expected exactly 12 frozen result files")

    # Scan anonymous package after sanitization.
    for p in supp.rglob("*"):
        if IDENT_RE.search(p.name): raise RuntimeError(f"identifying filename in supplement: {p}")
        if p.is_file():
            try: text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError: continue
            if IDENT_RE.search(text): raise RuntimeError(f"identifying string in supplement: {p}")

    sums = []
    for p in sorted(x for x in supp.rglob("*") if x.is_file() and x.name != "SHA256SUMS.txt"):
        sums.append(f"{sha256(p)}  {p.relative_to(supp).as_posix()}")
    (supp / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return supp


def zip_dir(src: Path, dest: Path, include_root=True):
    if dest.exists(): dest.unlink()
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        base = src.parent if include_root else src
        for p in sorted(x for x in src.rglob("*") if x.is_file()):
            z.write(p, p.relative_to(base).as_posix())


def extracted_verify(supp_zip: Path, diagnostics: Path, work: Path):
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(supp_zip) as z: z.extractall(work)
    supp = work / "anonymous_supplement"
    run(["sha256sum","-c","SHA256SUMS.txt"], cwd=supp)
    run([sys.executable,"scripts/build_locked_v07_reviewer_controls.py","--out",str(work/"v07"),"--per-family","80"], cwd=supp)
    rebuilt = work / "v07/items.jsonl"
    if sha256(rebuilt) != V07_HASH: raise RuntimeError("extracted supplement v0.7 hash mismatch")
    run([sys.executable,"scripts/build_v05_corpus.py","--out",str(work/"v05"),"--per-family","400"], cwd=supp)
    if (supp/"experiments/assistant_v05_seed202/report.json").read_bytes() != (work/"v05/report.json").read_bytes(): raise RuntimeError("v0.5 report mismatch")
    if (supp/"experiments/assistant_v05_seed202/sha256.json").read_bytes() != (work/"v05/sha256.json").read_bytes(): raise RuntimeError("v0.5 sha manifest mismatch")
    run([sys.executable,"scripts/v07_contrasts.py","--root","eval/locked_v07_reviewer_controls/outputs","--json-output",str(work/"contrasts.json"),"--text-output",str(work/"contrasts.txt"),"--bootstrap","10000","--seed","70917"], cwd=supp)
    if (supp/"eval/locked_v07_reviewer_controls/CONTRASTS_HIERARCHICAL.json").read_bytes() != (work/"contrasts.json").read_bytes(): raise RuntimeError("contrast JSON mismatch")
    if (supp/"eval/locked_v07_reviewer_controls/CONTRASTS.txt").read_bytes() != (work/"contrasts.txt").read_bytes(): raise RuntimeError("contrast text mismatch")
    run([sys.executable,"scripts/analyze_v07_submission_diagnostics.py","--root","eval/locked_v07_reviewer_controls/outputs","--out-dir",str(work/"diag"),"--bootstrap","10000","--seed","20260917"], cwd=supp)
    for name in ("submission_diagnostics.json","absolute_performance.csv","switch_rates.csv","seed_sensitivity.csv","SUBMISSION_DIAGNOSTICS.md"):
        if (supp/f"eval/locked_v07_reviewer_controls/submission_diagnostics/{name}").read_bytes() != (work/"diag"/name).read_bytes(): raise RuntimeError(f"diagnostics mismatch: {name}")
    return supp


def page_number_for(pdf: Path, needle: str) -> int | None:
    tmp = pdf.parent / "_pages"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    run(["pdftotext","-f","1","-l","999",str(pdf),str(tmp/"all.txt")])
    # Use form-feed page delimiters emitted by pdftotext.
    text = (tmp/"all.txt").read_text(errors="ignore")
    pages = text.split("\f")
    for idx, page in enumerate(pages, 1):
        if needle.lower() in page.lower(): return idx
    return None


def make_fields(dist: Path):
    content = (ROOT / "paper/openreview_abstract.txt").read_text(encoding="utf-8").strip().split("\n\n",1)
    title, abstract = content[0], content[1]
    text = f"""TITLE
{title}

KEYWORDS
large language models, model identity, self-binding, history weighting, behavioral evaluation, counterfactual evaluation, post-training, model behavior

TL;DR
Matched SELF, OTHER, and role-shifted evaluation-only FOCAL controls show that SELF-OTHER history sensitivity can contain separable operational components whose balance changes after mixed training, with substantial task-family and checkpoint heterogeneity.

ABSTRACT
{abstract}

PRIMARY AREA
foundation or frontier models, including LLMs

AI ASSISTANCE CATEGORIES SUPPORTED BY THE DISCLOSURE
- Yes, to aid or polish writing. Details are described in the paper.
- Yes, for retrieval and discovery (e.g., finding related work). Details are described in the paper.
- Yes, for research ideation or execution. Details are described in the paper.
- Yes, to draft sections of the paper. Details are described in the paper.
- Yes, for generating synthetic datasets. Details are described in the paper. (AI assisted authoring/revising deterministic dataset-template/generator code; no external model provider instantiated, calibrated, filtered, or selected the frozen v0.7 benchmark.)

AUTHOR-SIDE FIELDS NOT CHANGED HERE
Reciprocal-reviewing author/exemption, ethics/visibility/submission attestations, license, and final OpenReview upload remain author actions.
"""
    path = dist / "iclr2027_openreview_fields_revised.txt"
    path.write_text(text, encoding="utf-8")
    return path


def make_source_zip(dist: Path, commit: str):
    srcroot = dist / "source_revised"
    if srcroot.exists(): shutil.rmtree(srcroot)
    srcroot.mkdir()
    for f in ["paper/main.tex","paper/references.bib","paper/iclr2027_conference.sty","paper/iclr2027_conference.bst","paper/math_commands.tex","paper/openreview_abstract.txt","paper/SUBMISSION_CHECKLIST.md","scripts/apply_iclr2027_revision.py","scripts/analyze_v07_submission_diagnostics.py","scripts/build_iclr2027_revision_release.py"]:
        copy_file(f, srcroot / f)
    (srcroot / "README_BUILD.md").write_text(f"Private editable author archive for commit `{commit}`. Build from `paper/` with `pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex; pdflatex main.tex`. This archive is not the anonymous supplementary submission.\n", encoding="utf-8")
    dest = dist / "arewethesame_iclr2027_source_revised.zip"
    zip_dir(srcroot, dest, include_root=False)
    return dest


def make_report(dist: Path, commit: str, pdf: Path, supp_zip: Path, source_zip: Path, fields: Path, diagnostics: Path):
    info = run(["pdfinfo",str(pdf)], capture=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", info, re.M)
    pages = int(m.group(1)) if m else -1
    conclusion = page_number_for(pdf, "Conclusion")
    refs = page_number_for(pdf, "References")
    appendix = page_number_for(pdf, "Pilot progression and design corrections")
    ai = page_number_for(pdf, "AI use statement")
    repro = page_number_for(pdf, "Reproducibility statement")
    if conclusion is None or conclusion > 9: raise RuntimeError(f"counted main text exceeds 9 pages or Conclusion not found: {conclusion}")
    diag = json.loads((diagnostics/"submission_diagnostics.json").read_text())
    artifacts = [(pdf.name,pdf),(supp_zip.name,supp_zip),(source_zip.name,source_zip),(fields.name,fields)]
    rows = "\n".join(f"| `{name}` | {p.stat().st_size} | `{sha256(p)}` |" for name,p in artifacts)
    report = f"""# ICLR 2027 revision and verification report

## Revision identity

- Existing OpenReview submission: NVKUgJTxdI
- Branch: `paper/iclr2027-draft`
- Final source commit: `{commit}`
- GPU experiments/model inference: **not run**; frozen experimental evidence and preregistrations preserved.
- Frozen v0.7 items SHA-256: `{V07_HASH}`.

## Substantive revision

The paper now defines all four v0.7 explicit-policy families and their template structure; gives complete task/header examples; defines eligible-token option normalization, designated counterfactual options, S, current-facts margin/entropy, and irrelevant-metadata sensitivity exactly as implemented; documents recoverable training/evaluation settings and archival gaps; explicitly treats SELF-FOCAL as an operational residual because FOCAL is role-shifted, differently worded, evaluation-only, and exposure-mismatched; removes 95%-CI significance stars; and adds bounded post-hoc CPU diagnostics from the frozen outputs only.

The preregistered primary estimates were not recomputed from new model inference and were not altered. The authoritative stored contrast files remain the source of the reported primary intervals.

## Exploratory frozen-output diagnostics

- Mixed Qwen designated score SELF/OTHER/FOCAL: {diag['qwen']['mixed']['absolute_performance']['self']['mean_designated_score']:.4f} / {diag['qwen']['mixed']['absolute_performance']['other']['mean_designated_score']:.4f} / {diag['qwen']['mixed']['absolute_performance']['focal']['mean_designated_score']:.4f}.
- Mixed Qwen hard policy agreement SELF/OTHER/FOCAL: {diag['qwen']['mixed']['absolute_performance']['self']['policy_agreement_accuracy']:.4f} / {diag['qwen']['mixed']['absolute_performance']['other']['policy_agreement_accuracy']:.4f} / {diag['qwen']['mixed']['absolute_performance']['focal']['policy_agreement_accuracy']:.4f}.
- Mixed Qwen mean S SELF/OTHER/FOCAL: {diag['qwen']['mixed']['absolute_performance']['self']['mean_S_sensitivity']:.4f} / {diag['qwen']['mixed']['absolute_performance']['other']['mean_S_sensitivity']:.4f} / {diag['qwen']['mixed']['absolute_performance']['focal']['mean_S_sensitivity']:.4f}.
- Mixed Qwen matched switch rates: SELF-OTHER {diag['qwen']['mixed']['switch_rates']['self_vs_other']['switch_rate']:.4%}; SELF-FOCAL {diag['qwen']['mixed']['switch_rates']['self_vs_focal']['switch_rate']:.4%}; descriptive denominator 6,400 variants each.
- Qwen leave-one-training-seed-out SELF-FOCAL change range: {diag['qwen']['seed_sensitivity']['loo_ranges']['self_focal_change_vs_base']['min']:+.4f} to {diag['qwen']['seed_sensitivity']['loo_ranges']['self_focal_change_vs_base']['max']:+.4f}; FOCAL-OTHER change: {diag['qwen']['seed_sensitivity']['loo_ranges']['focal_other_change_vs_base']['min']:+.4f} to {diag['qwen']['seed_sensitivity']['loo_ranges']['focal_other_change_vs_base']['max']:+.4f}.
- Mistral leave-one-training-seed-out mixed ownership changes sign across omissions, while its SELF-FOCAL change remains negative and FOCAL-OTHER change remains positive.
- Eligible-token mass vs legacy max stored normalized probabilities: Qwen maximum absolute difference {diag['scoring_convention_check']['qwen']['max_abs_probability_difference_mass_vs_max']:.3e}; Mistral {diag['scoring_convention_check']['mistral']['max_abs_probability_difference_mass_vs_max']:.3e}. The legacy comparison is therefore a scoring-convention check, not an independent experimental replication.

## Extracted-ZIP reproduction gates

- PASS: package checksum verification from the extracted anonymous supplement root.
- PASS: v0.7 regeneration returns 320 items / 80 per family and exact frozen item SHA-256.
- PASS: v0.5 regeneration using packaged `build_v04_corpus.py` + `build_v05_corpus.py` exactly matches frozen `report.json` and `sha256.json`.
- PASS: all 12 Qwen/Mistral frozen output JSONs are present, contain the same 320 item IDs and required scoring/metric records, and are byte-identical to repository frozen outputs.
- PASS: `v07_contrasts.py` rerun with 10,000 draws / RNG seed 70917 byte-matches both frozen authoritative JSON and readable text output.
- PASS: exploratory diagnostics rerun with 10,000 draws / RNG seed 20260917 byte-match packaged JSON/CSV/Markdown outputs.
- PASS: anonymous supplement scan contains no author/development-repository identifiers; scientific model identifiers are retained.

## PDF structure and build

- Total PDF pages: {pages}
- Conclusion page: {conclusion} (counted main text therefore within the 9-page limit)
- AI-use statement page: {ai}
- Reproducibility statement page: {repro}
- References first page: {refs}
- Appendices first page: {appendix}
- PDF build used the repository ICLR 2027 style and BibTeX. CI rejects undefined citations/references and overfull boxes; final manual visual inspection is recorded separately by the delivery agent after rendering every page.
- PDF metadata has an empty Author field and the release workflow scans rendered text/bytes for author/repository identifiers.

## Delivered artifact hashes

| Artifact | Bytes | SHA-256 |
|---|---:|---|
{rows}

## PAT-feedback disposition matrix (private author handoff)

| Concern | Disposition | Revision/evidence | Mode |
|---|---|---|---|
| Undefined task families/examples | Addressed | Main family table + complete capacity example; appendix and `TASK_EXAMPLES.md` give all four exact examples | Writing + frozen generator |
| Relationship of 7 training vs 4 evaluation families | Addressed conservatively | States conceptual overlap, not literal subset, and no invented selection rationale | Writing/code inspection |
| Control metrics unspecified | Addressed | Exact option-normalized score, margin, entropy and irrelevant-metadata equations | Writing/evaluator inspection |
| c0/c1 and third option ambiguous | Addressed | Designated H0/H1 letters defined; third option retained in A/B/C normalization and policy metrics | Writing/evaluator inspection |
| FOCAL role shift/exposure mismatch | Addressed as limitation, not "fixed" | Abstract/methods/discussion/limitations explicitly call SELF-FOCAL operational residual | Writing/future work |
| Missing fine-tuning settings | Addressed where evidence exists | Main + appendix settings; exact unarchived model revisions/GPU package versions disclosed | Source/runner inspection |
| Duplicate S equation | Addressed | One labeled definition; chronology cross-references it | Writing |
| Five-seed uncertainty/outlier | Addressed | Explicit five-run limitation + leave-one-seed-out diagnostics; all five seeds retained | Existing-output CPU analysis |
| Absolute performance absent | Addressed | Score/accuracy/S table and machine-readable diagnostics | Existing-output CPU analysis |
| Decision-level impact unclear | Addressed | Matched choice-switch rates with denominator | Existing-output CPU analysis |
| 95% CI triple-star notation | Addressed | Significance stars removed | Presentation |
| Scoring robustness overclaimed | Addressed | Eligible-token signatures and mass-vs-max numerical identity/near-identity reported as scoring check | Existing-output inspection |
| Cross-model recipe parity confound | Addressed as limitation | Shared recipe may fit checkpoints differently; no architectural causal claim | Writing |
| Long-memory/frontier/self-recognition extensions | Scoped out | Listed as external-validity/future work; no new experiments | Future work |

## Genuine unresolved limitations

FOCAL remains role-shifted and exposure-mismatched; only five independent mixed-training runs exist per checkpoint; four synthetic task families are narrow and evidence-quality dominates the aggregate; exact original GPU package versions and exact model/tokenizer revision hashes were not archived; and the shared training recipe may fit checkpoints differently. These are disclosed rather than repaired with new experiments.

## Remaining author-side OpenReview actions

For existing submission NVKUgJTxdI: upload `{pdf.name}` to PDF; replace Supplementary Material with `{supp_zip.name}`; copy/synchronize title/TL;DR/abstract/keywords from `{fields.name}` as appropriate; review the AI-assistance boxes; and personally complete reciprocal-review, ethics, visibility, submission, and license attestations. No OpenReview upload or author attestation was performed by this workflow.
"""
    path = dist / "iclr2027_revision_and_verification_report.md"
    path.write_text(report, encoding="utf-8")
    page = {"total_pages":pages,"conclusion_page":conclusion,"ai_use_page":ai,"reproducibility_page":repro,"references_first_page":refs,"appendix_first_page":appendix}
    (dist/"page_structure.json").write_text(json.dumps(page,indent=2), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=ROOT/"dist_iclr_revision")
    ap.add_argument("--diagnostics", type=Path, default=ROOT/"revision_diagnostics")
    ap.add_argument("--v07-rebuilt", type=Path, required=True)
    ap.add_argument("--commit", default=os.environ.get("GITHUB_SHA", "unknown"))
    args = ap.parse_args()
    dist = args.dist.resolve(); diagnostics = args.diagnostics.resolve(); v07 = args.v07_rebuilt.resolve()
    if dist.exists(): shutil.rmtree(dist)
    dist.mkdir(parents=True)
    supp = assemble_supplement(dist, diagnostics, v07)
    supp_zip = dist / "arewethesame_iclr2027_anonymous_supplement_revised.zip"
    zip_dir(supp, supp_zip, include_root=True)
    if supp_zip.stat().st_size >= 100*1024*1024: raise RuntimeError("supplement exceeds 100MB")
    extracted_verify(supp_zip, diagnostics, dist/"_fresh_extract")
    shutil.rmtree(dist/"_fresh_extract")
    pdf = dist / "arewethesame_iclr2027_submission_revised.pdf"
    copy_file("paper/main.pdf", pdf)
    if pdf.stat().st_size >= 50*1024*1024: raise RuntimeError("PDF exceeds 50MB")
    fields = make_fields(dist)
    source_zip = make_source_zip(dist, args.commit)
    make_report(dist, args.commit, pdf, supp_zip, source_zip, fields, diagnostics)
    # Anonymous PDF text/byte scan.
    text = run(["pdftotext",str(pdf),"-"], capture=True).stdout
    if IDENT_RE.search(text): raise RuntimeError("identifying string in PDF text")
    strings = run(["strings",str(pdf)], capture=True).stdout
    if IDENT_RE.search(strings): raise RuntimeError("identifying string in PDF bytes")
    print("Release package complete:")
    for p in sorted(dist.iterdir()):
        if p.is_file(): print(p.name, p.stat().st_size, sha256(p))

if __name__ == "__main__":
    main()
