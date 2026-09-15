#!/usr/bin/env python3
"""Finalize the ICLR v0.7 manuscript after the crossed hierarchical audit.

This script performs no model inference. It synchronizes paper claims to the frozen
v0.7 evidence and the corrected seed x item hierarchical bootstrap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.tex"
ABSTRACT_FILE = ROOT / "paper" / "openreview_abstract.txt"
RESULTS_MD = ROOT / "eval" / "locked_v07_reviewer_controls" / "RESULTS_v07.md"
AUDIT_MD = ROOT / "docs" / "FINAL_STATISTICAL_AUDIT_V07.md"
HIER = ROOT / "eval" / "locked_v07_reviewer_controls" / "CONTRASTS_HIERARCHICAL.json"


def section_replace(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


def assert_close(x: float, expected: float, tol: float = 5e-4) -> None:
    if abs(x - expected) > tol:
        raise RuntimeError(f"authoritative result mismatch: {x} vs {expected}")


def main() -> None:
    h = json.loads(HIER.read_text())
    q = h["architectures"]["qwen"]
    m = h["architectures"]["mistral"]
    cross = h["qwen_minus_mistral"]
    loo = h["leave_one_out_evidence_quality"]

    # Fail closed if these are not the audited results used below.
    assert_close(q["mixed"]["delta_ownership_sensitivity"]["mean"], 0.7296875)
    assert_close(q["mixed"]["delta_self_vs_focal_sensitivity"]["mean"], -0.0482422)
    assert_close(q["mixed_minus_base"]["delta_self_vs_focal_sensitivity"]["mean"], -0.3328125)
    assert_close(cross["mixed"]["delta_self_vs_focal_sensitivity"]["mean"], 0.1105274)

    text = PAPER.read_text(encoding="utf-8")

    abstract = r"""\begin{abstract}
When a language model treats a history as its own, does that history receive privileged behavioral influence beyond the influence of an equally task-relevant non-self entity? We study this question with matched history-dependent forced-choice tasks whose designated actions follow explicit simulator policies. A 320-item preregistered reviewer-control benchmark holds the historical body fixed while crossing assigned identity with header order and adding a FOCAL condition: a task-relevant agent that is explicitly not the assistant and is introduced only at evaluation. Base Qwen 4B shows a SELF--OTHER sensitivity of $+0.72$ (95\% CI $[+0.57,+0.86]$), decomposing descriptively into SELF--FOCAL $+0.28$ $[+0.17,+0.41]$ and FOCAL--OTHER $+0.43$ $[+0.25,+0.61]$. After existing SELF/OTHER/NEUTRAL mixed training, total SELF--OTHER sensitivity is essentially unchanged at $+0.73$ $[+0.58,+0.90]$, but the aggregate SELF--FOCAL contrast is $-0.05$ $[-0.21,+0.16]$ while FOCAL--OTHER is $+0.78$ $[+0.50,+1.06]$. A crossed hierarchical bootstrap confirms a negative base-to-mixed change in SELF--FOCAL ($-0.33$ $[-0.54,-0.10]$) and a positive change in FOCAL--OTHER ($+0.35$ $[+0.05,+0.62]$), with no detectable change in total ownership. Mistral 7B behaves differently: its SELF--OTHER sensitivity falls from $+0.13$ to $-0.01$ after mixed training. Qwen and Mistral differ significantly in mixed SELF--OTHER sensitivity ($+0.74$ $[+0.58,+0.91]$), but not reliably in mixed SELF--FOCAL ($+0.11$ $[-0.07,+0.33]$). Current-facts-only and irrelevant-metadata controls make a simple uniform SELF-induced confidence-sharpening account less plausible, without identifying a unique internal mechanism. The main contribution is methodological: a SELF--OTHER contrast can combine self-specific, focal-agent, header-order, confidence, and family-specific effects, and these components must be separated before attributing behavioral privilege to self-binding.
\end{abstract}"""
    text = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", abstract, text, count=1, flags=re.S)

    matched = r"""\subsection{Matched history-dependent corpus}

A deterministic simulator generates 2,800 latent training pairs across seven history-dependent families. The v0.7 reviewer-control benchmark is separate: it contains 320 latent items, 80 each from capacity, reliability, horizon, and evidence-quality families. Every v0.7 item prints an explicit decision rule and mutually exclusive actions; the target is therefore a \emph{designated policy action}, not a claim of normative optimality. No model or external provider is used to generate, calibrate, filter, or select these items.

For v0.7, the historical body is byte-identical across SELF, OTHER, and FOCAL assignments. Only the identity header changes. Table~\ref{tab:matched-example} gives a schematic example using the actual header pattern.

\begin{table}[t]
\caption{Schematic v0.7 identity manipulation. The history body, current facts, policy, options, and designated action are identical across conditions; only the identity header changes. Header order is crossed separately.}
\label{tab:matched-example}
\small
\begin{center}
\begin{tabular}{p{0.13\linewidth}p{0.77\linewidth}}
\toprule
Condition & Header followed by the same history body \\
\midrule
\textsc{self} & ``Agent K is you, the assistant. Agent M is a different agent.'' \\
\textsc{other} & ``Agent K is a different agent. Agent M is you, the assistant.'' \\
\textsc{focal} & ``Agent K is the focal agent for this decision. Agent M is a different agent. You are an outside evaluator and are neither agent.'' \\
\midrule
History & ``Earlier, Agent K reserved 12 slots and has used 8. The remaining count is not precomputed. ...'' \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

The FOCAL condition is an \emph{evaluation-only} control: unlike SELF, OTHER, and NEUTRAL, it was not included in mixed-adapter training. It therefore tests generalization to a task-relevant non-self entity, but it is not perfectly exposure-matched to the trained bindings.

"""
    text = section_replace(text, r"\subsection{Matched history-dependent corpus}", r"\subsection{Ownership $\times$ header-order design}", matched)

    prereg = r"""\section{Preregistered Tests and Chronology}
\label{sec:prereg}

The experiments were developed sequentially, and we keep confirmatory and later reviewer-driven analyses distinct. In v0.5, the mixed adapter was trained on SELF, OTHER, and NEUTRAL bindings at approximately balanced frequency, with the designated target determined only by latent history. The v0.5 preregistered statistic was the within-model SELF-minus-OTHER difference in normalized designated-option score. V0.6 subsequently introduced byte-identical history bodies and assigned-identity aliases; its primary behavioral construct was an ownership-by-counterfactual-history log-odds interaction.

V0.7 is a separate, post-v0.6 reviewer-driven preregistered extension. It adds an evaluation-only FOCAL condition, crosses historical-owner position in the identity header, uses explicit printed policies, and adds current-facts-only and irrelevant-metadata controls. For condition $b$,
\begin{equation}
S_b = [\log P(c_0|H_0,b)-\log P(c_1|H_0,b)]-[\log P(c_0|H_1,b)-\log P(c_1|H_1,b)],
\end{equation}
where $H_0$ and $H_1$ are coherent histories implying different designated policy actions. V0.7 preregisters $\Delta_{\mathrm{ownership}}=\E[S_{\mathrm{self}}-S_{\mathrm{other}}]$ as the primary directional statistic and reports the decomposition
\begin{equation}
\Delta_{\mathrm{ownership}}=\Delta_{\mathrm{self\text{-}focal}}+\Delta_{\mathrm{focal\text{-}other}}.
\end{equation}
It also reports the header-order interaction and the confidence/irrelevant-evidence controls. Because FOCAL is evaluation-only, SELF--FOCAL is interpreted as a behavioral decomposition/control, not as a perfectly exposure-matched causal isolation of semantic selfhood.

All pooled mixed-adapter intervals in v0.7 use the preregistered crossed hierarchical bootstrap: training seeds and latent items are resampled with replacement while all within-item variants are preserved. Direct mixed Qwen--Mistral interactions resample the two seed sets independently and share the sampled item IDs. Base checkpoints have one realization per family and therefore use item-level bootstrap intervals.

"""
    text = section_replace(text, r"\section{Preregistered Self-Specificity Test}", r"\section{Results}", prereg)

    results = r"""\section{Results}
\label{sec:results}

\subsection{Earlier controls establish history use and motivate v0.7}

The earlier mixed-adapter experiments show that the models can strongly use coherent historical state: on the v0.5 Qwen evaluation, coherent-history designated-option scores are near one while shuffled-history scores are near chance. V0.6 then removes the SELF/OTHER history-text difference and shows that identity assignment can change counterfactual log-odds sensitivity even when mean policy agreement changes little. V0.7 asks whether that sensitivity is specifically SELF-related or partly attributable to a task-relevant focal entity and discourse structure.

\subsection{Base Qwen contains both SELF-over-FOCAL and FOCAL-over-OTHER components}

\begin{table}[t]
\caption{V0.7 Qwen decomposition. Mixed intervals and changes use the crossed seed$\times$item bootstrap. $***$ indicates a 95\% CI excluding zero.}
\label{tab:decomposition}
\small
\begin{center}
\begin{tabular}{lccc}
\toprule
Contrast & Base & Mixed & Mixed $-$ Base \\
\midrule
$\Delta_{\mathrm{ownership}}$ & $+.72\,[+.57,+.86]^{***}$ & $+.73\,[+.58,+.90]^{***}$ & $+.01\,[-.12,+.15]$ \\
$\Delta_{\mathrm{self\text{-}focal}}$ & $+.28\,[+.17,+.41]^{***}$ & $-.05\,[-.21,+.16]$ & $-.33\,[-.54,-.10]^{***}$ \\
$\Delta_{\mathrm{focal\text{-}other}}$ & $+.43\,[+.25,+.61]^{***}$ & $+.78\,[+.50,+1.06]^{***}$ & $+.35\,[+.05,+.62]^{***}$ \\
$\Delta_{\mathrm{order}}$ & $+.40\,[+.25,+.56]^{***}$ & $+.70\,[+.50,+.93]^{***}$ & $+.30\,[+.06,+.56]^{***}$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

For base Qwen, the SELF--OTHER interaction decomposes descriptively as $+.72\approx+.28+.43$: roughly 40\% SELF--FOCAL and 60\% FOCAL--OTHER. These percentages are descriptive ratios for this base checkpoint, not universal mixture weights.

After SELF/OTHER/NEUTRAL mixed training, total SELF--OTHER sensitivity is statistically unchanged. However, the aggregate SELF--FOCAL component decreases significantly and its mixed estimate spans zero, while FOCAL--OTHER increases significantly. Thus training changes the composition of the SELF--OTHER interaction without detectably increasing its total magnitude. This is the central v0.7 result. It does \emph{not} imply that every task family has zero SELF--FOCAL effect.

\subsection{The aggregate Qwen result is heterogeneous across families}

\begin{table}[t]
\caption{Qwen mixed-adapter family estimates under the hierarchical bootstrap.}
\label{tab:families}
\small
\begin{center}
\begin{tabular}{lcc}
\toprule
Family & $\Delta_{\mathrm{ownership}}$ & $\Delta_{\mathrm{self\text{-}focal}}$ \\
\midrule
capacity & $+.23\,[+.12,+.37]^{***}$ & $+.26\,[+.17,+.37]^{***}$ \\
evidence quality & $+2.57\,[+2.21,+2.91]^{***}$ & $-.36\,[-.75,+.11]$ \\
horizon & $+.05\,[-.03,+.13]$ & $+.16\,[+.04,+.32]^{***}$ \\
reliability & $+.08\,[-.23,+.39]$ & $-.25\,[-.54,+.07]$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

The aggregate SELF--FOCAL estimate therefore masks substantial family heterogeneity: capacity and horizon remain positive, whereas evidence-quality and reliability do not. Evidence-quality dominates the magnitude of the overall ownership statistic. Excluding that family leaves Qwen base ownership positive ($+.17\,[+.05,+.30]$), but mixed ownership becomes borderline/inconclusive ($+.12\,[-.00,+.23]$); mixed SELF--FOCAL is also inconclusive ($+.06\,[-.08,+.21]$). The leave-one-out analysis therefore supports dominance by evidence-quality more strongly than a claim of a uniform cross-family effect.

\subsection{Header order and confidence controls constrain alternative explanations}

The Qwen header-order interaction rises from $+.40\,[+.25,+.56]$ in base to $+.70\,[+.50,+.93]$ in mixed; the base-to-mixed change is $+.30\,[+.06,+.56]$. Identity assignment is therefore entangled with discourse position even after explicitly crossing order, and header salience should not be treated as negligible.

A simple alternative is uniform SELF-induced confidence sharpening. In base Qwen, relevant-history ownership sensitivity is positive ($+.72\,[+.57,+.86]$) while the current-facts-only SELF-minus-OTHER margin is negative ($-.39\,[-.54,-.25]$). In mixed Qwen, the control margin is near zero ($-.07\,[-.15,+.00]$) while ownership remains positive. These opposite/near-zero control patterns are inconsistent with the simplest global-sharpening account, although they do not eliminate all calibration or confidence mechanisms. Irrelevant-metadata sensitivity is small (base $-.04\,[-.08,-.00]$; mixed $-.02\,[-.03,+.00]$), providing little evidence that SELF indiscriminately amplifies arbitrary prompt perturbations.

\subsection{Mistral and direct checkpoint interactions}

Mistral behaves differently. Its ownership sensitivity changes from $+.13\,[+.05,+.21]$ in base to $-.01\,[-.09,+.07]$ in mixed (change $-.14\,[-.24,-.03]$). Its SELF--FOCAL contrast is $+.04\,[-.04,+.12]$ in base and $-.16\,[-.23,-.08]$ in mixed.

\begin{table}[t]
\caption{Direct Qwen-minus-Mistral interactions. These are checkpoint-family differences, not identified architectural causal effects.}
\label{tab:crossarch}
\small
\begin{center}
\begin{tabular}{lcc}
\toprule
Contrast & Base & Mixed \\
\midrule
$\Delta_{\mathrm{ownership}}$ & $+.59\,[+.46,+.71]^{***}$ & $+.74\,[+.58,+.91]^{***}$ \\
$\Delta_{\mathrm{self\text{-}focal}}$ & $+.24\,[+.08,+.40]^{***}$ & $+.11\,[-.07,+.33]$ \\
$\Delta_{\mathrm{focal\text{-}other}}$ & $+.35\,[+.17,+.53]^{***}$ & $+.63\,[+.35,+.92]^{***}$ \\
$\Delta_{\mathrm{order}}$ & $+.46\,[+.26,+.65]^{***}$ & $+.98\,[+.65,+1.35]^{***}$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

The mixed checkpoint-family difference is clear for total ownership, focal-vs-other, and order, but the mixed SELF--FOCAL difference is not reliably separated from zero. Qwen and Mistral also differ in architecture, pretraining, tokenization, size, and instruction tuning; these contrasts therefore identify checkpoint-family heterogeneity, not an architectural cause.

"""
    text = section_replace(text, r"\section{Results}", r"\section{Analysis and Discussion}", results)

    discussion = r"""\section{Analysis and Discussion}
\label{sec:discussion}

\textbf{The decomposition is the finding.}
A SELF--OTHER contrast alone is insufficient evidence for a privileged SELF representation. In base Qwen, both SELF--FOCAL and FOCAL--OTHER are positive. After mixed SELF/OTHER/NEUTRAL training, the aggregate SELF--FOCAL component decreases significantly and is no longer distinguishable from zero, while FOCAL--OTHER grows. This redistribution survives the preregistered crossed hierarchical bootstrap. The result is behavioral: it separates components of an identity-dependent log-odds interaction but does not identify an internal ``focalization mechanism.''

\textbf{Aggregate nulls can hide heterogeneous effects.}
The mixed Qwen SELF--FOCAL aggregate is near zero, but family estimates have different signs and capacity/horizon remain positive. We therefore do not interpret the aggregate as evidence that self-specific effects are absent on every task. Instead, v0.7 shows that the aggregate SELF--OTHER effect is compositionally heterogeneous and heavily influenced by evidence-quality items.

\textbf{The FOCAL control is informative but not exposure-matched.}
FOCAL was not present during mixed training. Its value is that it tests whether a task-relevant non-self entity can reproduce much of the SELF--OTHER sensitivity on an unseen binding. Its limitation is equally important: SELF--FOCAL can reflect both semantic ownership and generalization/novelty differences. It should therefore be read as a decomposition/control rather than a pure causal estimate of ``selfhood.''

\textbf{Confidence and discourse alternatives are narrowed, not eliminated.}
The current-facts-only control goes in the opposite direction from base Qwen ownership, making a simple uniform SELF-induced logit sharpening account implausible. The small irrelevant-metadata effects similarly argue against indiscriminate amplification of all prompt changes. At the same time, the strong header-order interaction shows that discourse salience remains behaviorally important. More general calibration transformations or representation-level mechanisms remain possible.

\textbf{Checkpoint specificity.}
Qwen and Mistral differ substantially on ownership and focal-agent effects, but not all direct interactions are significant. The mixed SELF--FOCAL Qwen-minus-Mistral interval spans zero. Because the checkpoints differ along many dimensions, we make no architectural causal claim.

\textbf{What the results do not establish.}
We do not claim that language models lack self-reference representations, that FOCAL and SELF are internally identical, or that the observed effects bear on subjective experience or consciousness. The experiments measure behavioral sensitivity under controlled prompt assignments and post-training interventions.

"""
    text = section_replace(text, r"\section{Analysis and Discussion}", r"\section{Limitations}", discussion)

    limitations = r"""\section{Limitations}

First, FOCAL is an evaluation-only control rather than an exposure-matched training condition; SELF--FOCAL therefore combines semantic-ownership differences with generalization to an unseen binding. Second, the v0.7 aggregate is heterogeneous across task families, and evidence-quality contributes disproportionately to the total effect. Third, the counterfactual log-odds statistic is behavioral and can reflect calibration or confidence changes as well as evidence use; our controls constrain the simplest alternatives but do not identify a unique internal mechanism. Fourth, the tasks are synthetic explicit-policy decisions, so the target measures agreement with a stated policy rather than normative decision quality. Fifth, our intervention is post-training only and uses 4B/7B instruction-tuned checkpoints rather than frontier-scale systems. Finally, forced-choice scoring is intentionally narrow; open-ended planning, social interaction, long-horizon agency, and mechanistic causal interventions remain external-validity tests.

"""
    text = section_replace(text, r"\section{Limitations}", r"\section{Conclusion}", limitations)

    conclusion = r"""\section{Conclusion}

Assigned identity can change how language models respond to counterfactual history, but a SELF--OTHER contrast does not by itself isolate self-specific privilege. In base Qwen, the interaction contains both SELF--FOCAL and FOCAL--OTHER components. After SELF/OTHER/NEUTRAL mixed training, total SELF--OTHER sensitivity remains nearly unchanged, while the aggregate SELF--FOCAL component decreases significantly and FOCAL--OTHER increases significantly. Under the hierarchical analysis, the mixed SELF--FOCAL estimate itself spans zero, and the family-level results are heterogeneous. Mistral shows a different checkpoint-level pattern.

The main contribution is therefore methodological: tests of model self-binding should separate self-specificity from task relevance/focality, discourse position, confidence, family composition, and training familiarity. Under those controls, the strongest conclusion is not that ``self is focality,'' but that much of the apparent SELF--OTHER effect can migrate into an unseen focal-agent control after matched history training.

"""
    text = section_replace(text, r"\section{Conclusion}", r"\subsection*{AI use statement}", conclusion)

    # Remove stale mechanism/"correct" language that survives outside replaced sections.
    text = text.replace("correct decision", "designated policy action")
    text = text.replace("correct targets", "designated policy targets")
    text = text.replace("correct option changes", "designated option changes")
    text = text.replace("the correct decision remains fixed", "the designated policy action remains fixed")

    # Update appendix seed tables using frozen per-seed means.
    appendix = r"""\section{Per-seed replication results}

\begin{table}[h]
\caption{Qwen 4B mixed-adapter v0.7 seed means. The aggregate SELF--FOCAL estimate is heterogeneous across seeds.}
\small
\begin{center}
\begin{tabular}{lccc}
\toprule
Seed & $\Delta_{\mathrm{ownership}}$ & $\Delta_{\mathrm{self\text{-}focal}}$ & $\Delta_{\mathrm{focal\text{-}other}}$ \\
\midrule
31 & $+.700$ & $-.126$ & $+.826$ \\
42 & $+.616$ & $-.054$ & $+.670$ \\
73 & $+.766$ & $-.158$ & $+.925$ \\
128 & $+.900$ & $-.223$ & $+1.123$ \\
256 & $+.667$ & $+.320$ & $+.346$ \\
\midrule
Mean & $+.730$ & $-.048$ & $+.778$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\begin{table}[h]
\caption{Mistral 7B mixed-adapter v0.7 seed means.}
\small
\begin{center}
\begin{tabular}{lccc}
\toprule
Seed & $\Delta_{\mathrm{ownership}}$ & $\Delta_{\mathrm{self\text{-}focal}}$ & $\Delta_{\mathrm{focal\text{-}other}}$ \\
\midrule
31 & $+.112$ & $-.174$ & $+.286$ \\
42 & $-.028$ & $-.146$ & $+.118$ \\
73 & $-.140$ & $-.249$ & $+.109$ \\
128 & $+.027$ & $-.064$ & $+.091$ \\
256 & $-.009$ & $-.161$ & $+.153$ \\
\midrule
Mean & $-.008$ & $-.159$ & $+.151$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

"""
    text = section_replace(text, r"\section{Per-seed replication results}", r"\section{Additional artifacts}", appendix)

    PAPER.write_text(text, encoding="utf-8")

    title = "Does Assigned Identity Change How Language Models Use History? Separating Self-Binding from Focal-Agent Effects"
    abstract_plain = (
        "When a language model treats a history as its own, does that history receive privileged behavioral influence beyond the influence of an equally task-relevant non-self entity? "
        "We study this question with matched history-dependent forced-choice tasks whose designated actions follow explicit simulator policies. A 320-item preregistered reviewer-control benchmark holds the historical body fixed while crossing assigned identity with header order and adding a FOCAL condition: a task-relevant agent that is explicitly not the assistant and is introduced only at evaluation. "
        "Base Qwen 4B shows a SELF-OTHER sensitivity of +0.72 (95% CI [+0.57,+0.86]), decomposing descriptively into SELF-FOCAL +0.28 [+0.17,+0.41] and FOCAL-OTHER +0.43 [+0.25,+0.61]. After existing SELF/OTHER/NEUTRAL mixed training, total SELF-OTHER sensitivity is essentially unchanged at +0.73 [+0.58,+0.90], but the aggregate SELF-FOCAL contrast is -0.05 [-0.21,+0.16] while FOCAL-OTHER is +0.78 [+0.50,+1.06]. "
        "A crossed hierarchical bootstrap confirms a negative base-to-mixed change in SELF-FOCAL (-0.33 [-0.54,-0.10]) and a positive change in FOCAL-OTHER (+0.35 [+0.05,+0.62]), with no detectable change in total ownership. Mistral 7B behaves differently: its SELF-OTHER sensitivity falls from +0.13 to -0.01 after mixed training. Qwen and Mistral differ significantly in mixed SELF-OTHER sensitivity (+0.74 [+0.58,+0.91]), but not reliably in mixed SELF-FOCAL (+0.11 [-0.07,+0.33]). "
        "Current-facts-only and irrelevant-metadata controls make a simple uniform SELF-induced confidence-sharpening account less plausible, without identifying a unique internal mechanism. The main contribution is methodological: a SELF-OTHER contrast can combine self-specific, focal-agent, header-order, confidence, and family-specific effects, and these components must be separated before attributing behavioral privilege to self-binding."
    )
    ABSTRACT_FILE.write_text(title + "\n\n" + abstract_plain + "\n", encoding="utf-8")

    RESULTS_MD.write_text(r"""# V0.7 Results: Reviewer Controls — Hierarchical Final

**Status:** Complete. All 12 jobs finished. Primary inference below uses the preregistered crossed bootstrap over training seeds and latent items for mixed-adapter quantities.

## Qwen decomposition

| Contrast | Base | Mixed | Mixed - Base |
|---|---:|---:|---:|
| ownership (SELF-OTHER) | +0.715 [+0.571,+0.861] | +0.730 [+0.576,+0.897] | +0.015 [-0.118,+0.149] |
| SELF-FOCAL | +0.285 [+0.166,+0.406] | -0.048 [-0.214,+0.159] | -0.333 [-0.544,-0.098] |
| FOCAL-OTHER | +0.431 [+0.250,+0.608] | +0.778 [+0.500,+1.057] | +0.347 [+0.049,+0.620] |
| header-order interaction | +0.402 [+0.251,+0.560] | +0.704 [+0.498,+0.934] | +0.303 [+0.060,+0.565] |

Base Qwen therefore contains both a SELF-over-FOCAL and a FOCAL-over-OTHER component. After SELF/OTHER/NEUTRAL mixed training, total ownership is unchanged, while the aggregate SELF-FOCAL component decreases significantly and FOCAL-OTHER increases significantly. FOCAL is evaluation-only and was not exposure-matched during training, so this decomposition is a behavioral control, not a pure causal isolation of semantic selfhood.

## Mistral

| Contrast | Base | Mixed | Mixed - Base |
|---|---:|---:|---:|
| ownership | +0.128 [+0.046,+0.209] | -0.008 [-0.087,+0.065] | -0.135 [-0.242,-0.033] |
| SELF-FOCAL | +0.044 [-0.035,+0.121] | -0.159 [-0.231,-0.084] | -0.203 [-0.301,-0.106] |
| FOCAL-OTHER | +0.084 [+0.010,+0.160] | +0.151 [+0.077,+0.236] | +0.068 [-0.040,+0.180] |

## Direct Qwen - Mistral interactions

| Contrast | Base | Mixed |
|---|---:|---:|
| ownership | +0.588 [+0.465,+0.708] | +0.737 [+0.583,+0.906] |
| SELF-FOCAL | +0.240 [+0.076,+0.399] | +0.111 [-0.072,+0.335] |
| FOCAL-OTHER | +0.347 [+0.174,+0.529] | +0.627 [+0.345,+0.920] |
| header-order interaction | +0.458 [+0.264,+0.653] | +0.981 [+0.653,+1.352] |

The mixed SELF-FOCAL checkpoint difference is not reliably different from zero. Qwen and Mistral differ along multiple dimensions, so these are checkpoint-family interactions rather than architectural causal effects.

## Controls and heterogeneity

- Qwen base relevant-history ownership is positive (+0.715), while the history-irrelevant control margin is negative (-0.391 [-0.542,-0.246]); mixed control margin is near zero (-0.071 [-0.146,+0.003]). This is evidence against the simplest uniform SELF-induced confidence-sharpening account, not proof against every confidence mechanism.
- Qwen irrelevant-metadata sensitivity is small: base -0.042 [-0.084,-0.000] and mixed -0.016 [-0.033,+0.001].
- Excluding `evidence_quality`, Qwen ownership is +0.170 [+0.049,+0.295] in base but +0.117 [-0.001,+0.228] in mixed. Thus evidence-quality is a disproportionate driver and the mixed residual outside that family is inconclusive.
- Qwen mixed SELF-FOCAL is heterogeneous by family: capacity +0.259 [+0.174,+0.367], evidence_quality -0.360 [-0.752,+0.106], horizon +0.159 [+0.038,+0.322], reliability -0.251 [-0.541,+0.073]. Aggregate near-zero SELF-FOCAL therefore does not imply every family is null.

## Authoritative artifacts

- `outputs/summary.json`: preregistered hierarchical aggregate from the v0.7 evaluator.
- `CONTRASTS_HIERARCHICAL.json`: corrected 10,000-draw hierarchical contrasts, including mixed-minus-base and leave-one-out analyses.
- `CONTRASTS.txt`: human-readable rendering of the corrected hierarchical contrasts.
- `scripts/v07_contrasts.py`: reproducible CPU-only analysis; no model inference.
""", encoding="utf-8")

    audit = AUDIT_MD.read_text(encoding="utf-8")
    audit = re.sub(r"\*\*Status:\*\*.*?\n", "**Status:** RESOLVED. Corrected crossed-hierarchical contrasts were regenerated from the frozen per-item outputs and committed; manuscript claims must use `CONTRASTS_HIERARCHICAL.json` / `outputs/summary.json`.\n", audit, count=1)
    AUDIT_MD.write_text(audit, encoding="utf-8")

    print("Final v0.7 manuscript/evidence sync complete.")


if __name__ == "__main__":
    main()
