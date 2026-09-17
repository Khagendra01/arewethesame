from __future__ import annotations

from pathlib import Path

MAIN = Path('paper/main.tex')
REFS = Path('paper/references.bib')
OPENREVIEW = Path('paper/openreview_abstract.txt')
RESULTS = Path('eval/locked_v07_reviewer_controls/RESULTS_v07.md')
CHECKLIST = Path('paper/SUBMISSION_CHECKLIST.md')

TITLE = 'Does Assigned Identity Change How Language Models Use History? Separating Self-Binding from Focal-Agent Effects'

ABSTRACT_TEX = r'''When a language model treats a history as its own, does that history receive privileged behavioral influence beyond the influence of a task-relevant non-self entity under matched prompts? We study this question with forced-choice tasks whose designated actions follow explicit simulator policies. A 320-item preregistered reviewer-control benchmark holds each history body fixed while crossing assigned identity with header order and adding an evaluation-only FOCAL condition. FOCAL is task-relevant but also changes the assistant to an outside-evaluator role, so SELF--FOCAL is an operational residual rather than a causal isolation of selfhood. Base Qwen 4B shows SELF--OTHER sensitivity $+0.72$ (95\% CI $[+0.57,+0.86]$), with SELF--FOCAL $+0.28$ $[+0.17,+0.41]$ and FOCAL--OTHER $+0.43$ $[+0.25,+0.61]$. After SELF/OTHER/NEUTRAL mixed training, SELF--OTHER is $+0.73$ $[+0.58,+0.90]$ and its base-to-mixed change is $+0.01$ $[-0.12,+0.15]$, while SELF--FOCAL changes by $-0.33$ $[-0.54,-0.10]$ and FOCAL--OTHER by $+0.35$ $[+0.05,+0.62]$. Mistral 7B shows a different checkpoint-level pattern. Exploratory analyses of the frozen outputs show that the Qwen decomposition changes are sign-stable to leaving out any one training seed, but the study still has only five independent mixed-training runs. Current-facts and irrelevant-metadata controls narrow simple calibration explanations without identifying a unique mechanism. The methodological result is that SELF--OTHER alone is insufficient: role framing, focality/task relevance, header position, training exposure, and task-family composition must be separated before attributing behavioral privilege to self-binding.'''

INTRO_BLOCK = r'''The v0.7 extension adds a \textbf{FOCAL control}: the historical owner is task-relevant but explicitly not the assistant. Algebraically,
\begin{equation}
\label{eq:decomp}
\Delta_{\mathrm{ownership}}=\Delta_{\mathrm{self\text{-}focal}}+\Delta_{\mathrm{focal\text{-}other}}.
\end{equation}
The identity is exact by definition; the contribution is the matched intervention and the measured pattern. Importantly, FOCAL also instructs the assistant to act as an outside evaluator, uses different wording, and was absent from mixed training. We therefore treat SELF--FOCAL as an \emph{operational residual}, not a pure causal estimate of semantic selfhood.

Base Qwen shows positive SELF--FOCAL and FOCAL--OTHER components. After mixed SELF/OTHER/NEUTRAL training, we detect no change in total SELF--OTHER sensitivity, while the aggregate SELF--FOCAL component decreases and FOCAL--OTHER increases. Capacity and horizon nevertheless retain positive mixed SELF--FOCAL estimates, and Mistral behaves differently. The result is therefore a bounded behavioral decomposition, not evidence that ``self is focality'' or that any internal mechanism has been identified.

Our contributions are:
\begin{itemize}
    \item a controlled ownership/header-order benchmark that adds an evaluation-only focal non-self condition and explicit confidence/irrelevant-metadata controls;
    \item a precise decomposition of the observed SELF--OTHER log-odds interaction into operational SELF--FOCAL and FOCAL--OTHER contrasts, with role/exposure caveats made explicit;
    \item cross-checkpoint and task-family analyses showing substantial heterogeneity rather than a universal self-specific effect; and
    \item a reproducible frozen-output analysis package, including absolute policy performance, matched decision-switch rates, and leave-one-training-seed-out diagnostics.
\end{itemize}

'''

FRAMEWORK = r'''\section{Experimental Framework}
\label{sec:framework}

\subsection{Training corpus and explicit-policy evaluation}

The v0.5 training corpus contains 2,800 deterministic latent pairs across seven broad history-dependent families: resource allocation, exploration, cooperation, delayed reward, belief revision, trust, and persistence. The mixed adapter sees one of SELF/OTHER/NEUTRAL per pair at approximately balanced frequency; targets depend on latent history, not binding. The v0.7 reviewer-control benchmark is generated separately and contains four explicit-policy templates with 80 deterministic instantiations each. These templates conceptually overlap with the training themes but are not literal held-out instances of the seven training templates, and no separate rationale for selecting exactly these four families was preregistered.

\begin{table}[t]
\caption{The four v0.7 explicit-policy families. H0/H1 differ only in the task-relevant historical state; administrative metadata is held fixed.}
\label{tab:families-def}
\scriptsize
\begin{center}
\begin{tabular}{p{0.16\linewidth}p{0.47\linewidth}p{0.27\linewidth}}
\toprule
Family & Printed policy & H0 / H1 boundary \\
\midrule
Capacity & Explore iff at least 4 slots remain; otherwise confirm; never postpone. & 4 versus 3 slots remain. \\
Reliability & Share iff helpful fraction is at least .75; otherwise keep private; never defer. & 6/8 versus 5/8 helpful. \\
Horizon & Invest iff at least 10 episodes remain; otherwise take immediate gain; never split. & 10 versus 9 episodes remain. \\
Evidence quality & Retain iff $<30\%$ are biased; otherwise recompute; never reverse solely from this record. & 2/10 versus 4/10 biased. \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

There are four task templates, ten aliases, three answer rotations, and 15 arithmetic-control configurations. Capacity and horizon each draw one of four totals; reliability fixes the 6/8 versus 5/8 counts while shuffling outcome order; evidence-quality uses the same 2/10 versus 4/10 contrast in all 80 items. Ordered alias pairs and administrative metadata are independently sampled per deterministic item, so the 80 rows per family are instantiations, not 80 distinct task types. The frozen 320-item file has SHA-256 \texttt{d4960832...cc7a}.

\begin{table}[t]
\caption{Complete capacity example (\texttt{v07\_capacity\_0000}); no required prompt content is omitted. The six headers are shown compactly as owner/foil order variants.}
\label{tab:complete-example}
\scriptsize
\begin{center}
\begin{tabular}{p{0.18\linewidth}p{0.72\linewidth}}
\toprule
Field & Text \\
\midrule
Policy & Explore iff at least 4 slots remain; otherwise confirm; never postpone. \\
H0 & Agent V reserved 13 slots and used 9; metadata: batch 46, room 5, note 0-589. \\
H1 & Agent V reserved 13 slots and used 10; same metadata. \\
Current/question & Both tests are feasible when permitted. Which action follows the stated rule? \\
Options/targets & A confirm; B explore; C postpone. H0$\to$B, H1$\to$A. \\
SELF headers & first: V is you; Z different. second: Z different; V is you. \\
OTHER headers & first: V different; Z is you. second: Z is you; V different. \\
FOCAL headers & first: V focal; Z different; you are an outside evaluator/neither. second: Z different; V focal; same outside-evaluator instruction. \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

For every item the historical body, policy, current situation, question, options, and designated targets are byte-identical across SELF, OTHER, and FOCAL; only the identity header changes. Header order is crossed. FOCAL is evaluation-only and differs additionally in role instruction, wording/length, and training exposure, so FOCAL--OTHER measures the combined effect of those prompt conditions rather than an identified causal effect of focality alone. Appendix~\ref{app:examples} gives complete examples for all four families.

\subsection{Training configuration}

The frozen runners train completion-only QLoRA adapters~\citep{dettmers2023qlora,hu2021lora} on \texttt{Qwen/Qwen3-4B-Instruct-2507} and replicate on \texttt{mistralai/Mistral-7B-Instruct-v0.3}. The invoked defaults are learning rate $10^{-4}$, per-device batch 4, gradient accumulation 4 (effective batch 16 on the one-L4 runner), 3 epochs, maximum length 512, paged AdamW 8-bit, cosine schedule with 0.05 warmup ratio, zero weight decay, gradient norm 1.0, and LoRA rank 16/alpha 32/dropout .05 on q/k/v/o/gate/up/down projections. Quantization is 4-bit NF4 with double quantization; compute dtype is selected at runtime as bfloat16 when supported, otherwise float16. Prompt tokens are masked and only the assistant option response contributes to loss. The best epoch checkpoint is selected by validation loss. Exact package versions and model revision hashes were not persisted by the original GPU runs; the available runner constraints and full settings are documented in Appendix~\ref{app:settings}.

The v0.5 pair split contains 2,257 training, 275 validation, and 268 test pairs; 40 shared instruction anchors are added only to training, yielding 2,297 training rows per condition. Mixed-binding counts over the 2,800 latent pairs are 946 SELF, 940 OTHER, and 914 NEUTRAL. Five independent mixed adapters use training seeds 31, 42, 73, 128, and 256 for each checkpoint family. FOCAL is never used in training.

\subsection{Option scoring and counterfactual sensitivity}

Let $i$ index a latent item, $h$ a history variant, $b$ an ownership/header-order condition, and $a\in\{A,B,C\}$ an option. After applying the checkpoint chat template, let $\mathcal{T}_a$ be the unique token IDs among the bare-letter and leading-space realizations of $a$ that tokenize to a single token. The primary option-normalized score is
\begin{equation}
\label{eq:ptilde}
\widetilde P(a\mid x)=\frac{\sum_{t\in\mathcal{T}_a} P(t\mid x)}{\sum_{a'\in\{A,B,C\}}\sum_{t\in\mathcal{T}_{a'}}P(t\mid x)}.
\end{equation}
These are normalized next-token option scores, not unrestricted generation probabilities. Duplicate token IDs are counted once. The preregistered legacy convention takes the maximum eligible-token log probability for each option before the same A/B/C normalization.

Let $c_{0i}$ and $c_{1i}$ be the distinct designated option letters under counterfactual histories $H_{0i}$ and $H_{1i}$; letters vary with answer rotation. Define
\begin{align}
\label{eq:r}
r_{i,b}(H)&=\log\widetilde P(c_{0i}\mid H,b)-\log\widetilde P(c_{1i}\mid H,b),\\
\label{eq:s}
S_{i,b}&=r_{i,b}(H_{0i})-r_{i,b}(H_{1i}).
\end{align}
The third option remains in the three-option normalization, policy score, and hard accuracy but is not an endpoint of this pairwise log-odds statistic. We average first/second header order within SELF, OTHER, and FOCAL before forming $\Delta_{\mathrm{ownership}}=E[S_{\mathrm{self}}-S_{\mathrm{other}}]$, $\Delta_{\mathrm{self\text{-}focal}}$, and $\Delta_{\mathrm{focal\text{-}other}}$. Items receive equal weight; mixed-model estimates additionally weight the five training seeds equally. The order interaction is $(S_{\mathrm{self,first}}-S_{\mathrm{other,first}})-(S_{\mathrm{self,second}}-S_{\mathrm{other,second}})$.

\subsection{Controls and uncertainty}

The current-facts control \emph{retains H0} in the prompt but declares it irrelevant and adds an arithmetic rule. For example, item \texttt{v07\_capacity\_0000} says ``choose Multiply iff $8\times1>8$, otherwise Add; never Wait,'' so its designated answer is Add (rotated to C). If $d_i$ is the designated control option,
\begin{equation}
\label{eq:margin}
M_{i,b}=\log\widetilde P(d_i\mid b)-\max_{a\neq d_i}\log\widetilde P(a\mid b),
\end{equation}
and $\Delta_{\mathrm{control\_margin}}$ is the order-averaged SELF-minus-OTHER contrast of $M$. Entropy is $-\sum_a\widetilde P(a)\log\widetilde P(a)$ and its contrast is defined analogously.

For irrelevant metadata, $J_{0i}$ and $J_{1i}$ preserve the same task-relevant state and designated action but change only administrative batch/room/note text. Using the same $c_{0i},c_{1i}$ pair from Eq.~\ref{eq:r}, $I_{i,b}=r_{i,b}(J_{0i})-r_{i,b}(J_{1i})$ and $\Delta_{\mathrm{irrelevant}}$ is the order-averaged SELF-minus-OTHER contrast. These controls constrain simple prompt-wide calibration accounts but do not identify confidence or mechanism uniquely.

Pooled mixed intervals use the preregistered crossed hierarchical resampling structure: training seeds and latent item IDs are resampled with replacement while all within-item variants stay paired. Base checkpoints use item bootstrap intervals. The preregistration specified 5,000 draws; reported final intervals use 10,000 draws to reduce Monte Carlo noise without changing estimands or the resampling structure. Ten thousand draws do not compensate for having only five independent training runs.

'''

CHRONOLOGY = r'''\section{Design Chronology}
\label{sec:prereg}

The experiments were developed sequentially. V0.3 exposed history-independent supervision and response-format failure; v0.4 made targets history-dependent but showed that separately trained SELF/OTHER adapters had near-symmetric own-binding advantages, motivating a within-model design. V0.5 trained one mixed adapter with approximately balanced SELF/OTHER/NEUTRAL exposure. V0.6 introduced byte-identical histories with assigned identity and the counterfactual statistic in Eq.~\ref{eq:s}. V0.7 is a separate reviewer-driven extension registered after v0.6 and before v0.7 evaluation; it adds FOCAL, header-order crossing, explicit policies, and the two controls above. Per-family and leave-one-family-out analyses were preregistered as exploratory. The absolute-performance, matched-switch, and leave-one-training-seed-out diagnostics added in this revision are explicitly post hoc/exploratory and use only the frozen per-item outputs. Appendix~\ref{app:pilots} summarizes the progression.

'''

RESULTS_BLOCK = r'''\section{Results}
\label{sec:results}

\subsection{Qwen decomposition}

\begin{table}[t]
\caption{Qwen v0.7 decomposition under primary eligible-token-mass scoring. Intervals are 95\% crossed seed$\times$item (mixed/change) or item-bootstrap (base) intervals; no significance-star convention is used.}
\label{tab:decomposition}
\scriptsize
\begin{center}
\begin{tabular}{lccc}
\toprule
Contrast & Base & Mixed & Mixed $-$ Base \\
\midrule
$\Delta_{\mathrm{ownership}}$ & $+.715\,[+.571,+.861]$ & $+.730\,[+.576,+.897]$ & $+.014\,[-.118,+.149]$ \\
$\Delta_{\mathrm{self\text{-}focal}}$ & $+.285\,[+.166,+.406]$ & $-.048\,[-.214,+.159]$ & $-.333\,[-.544,-.098]$ \\
$\Delta_{\mathrm{focal\text{-}other}}$ & $+.431\,[+.250,+.608]$ & $+.778\,[+.500,+1.057]$ & $+.347\,[+.049,+.620]$ \\
$\Delta_{\mathrm{order}}$ & $+.402\,[+.251,+.560]$ & $+.704\,[+.498,+.934]$ & $+.303\,[+.060,+.565]$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

Base Qwen has positive SELF--FOCAL and FOCAL--OTHER contrasts. After mixed training, we detect no base-to-mixed change in total SELF--OTHER sensitivity ($+.014\,[-.118,+.149]$), while SELF--FOCAL decreases and FOCAL--OTHER increases. Equation~\ref{eq:decomp} is an algebraic identity; the empirical result is the measured redistribution under these prompt conditions. The legacy max-token convention is not an independent replication: stored Qwen normalized probabilities differ from eligible-token-mass scores by at most $1.3\times10^{-8}$, while Mistral bare/space realizations map to the same token IDs and are exactly identical under the two conventions.

\subsection{Family heterogeneity and checkpoint specificity}

\begin{table}[t]
\caption{Qwen mixed-adapter family estimates under primary mass scoring; family/leave-one-family-out analyses are exploratory.}
\label{tab:families}
\scriptsize
\begin{center}
\begin{tabular}{lcc}
\toprule
Family & $\Delta_{\mathrm{ownership}}$ & $\Delta_{\mathrm{self\text{-}focal}}$ \\
\midrule
capacity & $+.23\,[+.12,+.37]$ & $+.26\,[+.17,+.37]$ \\
evidence quality & $+2.57\,[+2.21,+2.91]$ & $-.36\,[-.75,+.11]$ \\
horizon & $+.05\,[-.03,+.13]$ & $+.16\,[+.04,+.32]$ \\
reliability & $+.08\,[-.23,+.39]$ & $-.25\,[-.54,+.07]$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

Capacity and horizon retain positive mixed SELF--FOCAL estimates, whereas evidence-quality and reliability do not. Evidence-quality dominates the aggregate magnitude: excluding it gives Qwen mixed ownership $+.1169\,[-.0005,+.2282]$ and mixed SELF--FOCAL $+.0557\,[-.0754,+.2072]$. Thus the aggregate should not be read as a uniform cross-family effect.

Mistral ownership changes from $+.128\,[+.046,+.209]$ in base to $-.008\,[-.087,+.065]$ in mixed (change $-.135\,[-.242,-.033]$); mixed SELF--FOCAL is $-.159\,[-.231,-.084]$. The direct mixed Qwen-minus-Mistral ownership contrast is $+.737\,[+.583,+.906]$, but mixed SELF--FOCAL is $+.111\,[-.072,+.335]$. These are checkpoint-family differences, not architectural causes: checkpoint size, tokenizer, pretraining, instruction tuning, and suitability of the shared training recipe all differ or remain uncontrolled.

\subsection{Controls and post-hoc frozen-output diagnostics}

Qwen's current-facts SELF-minus-OTHER designated-answer margin is negative in base ($-.391\,[-.534,-.247]$) and near zero in mixed ($-.071\,[-.147,+.002]$); the corresponding entropy contrasts are $+.0195\,[+.0084,+.0307]$ and $+.0018\,[-.0053,+.0089]$. A negative correct-answer margin does not itself mean lower confidence---a model can sharpen a wrong answer---so these controls only show that the separate control prompts do not exhibit the same pattern as relevant-history sensitivity. Irrelevant-metadata sensitivity is small (base $-.042$, mixed $-.016$).

Exploratory absolute metrics and matched decision switches are reported in Appendix~\ref{app:diagnostics}. In mixed Qwen, mean designated-option score / hard policy agreement / mean $S$ are SELF $.649/.661/4.065$, OTHER $.635/.648/3.335$, and FOCAL $.657/.668/4.113$. SELF versus OTHER choices differ on $2.27\%$ of 6,400 matched item-history-order-seed variants; SELF versus FOCAL differs on $2.78\%$. These are descriptive condition-induced switches, not responses to changing H0 into H1.

Leave-one-training-seed-out point estimates are also sign-stable for the two central Qwen changes: omitting any one seed leaves SELF--FOCAL mixed-minus-base between $-.425$ and $-.289$ and FOCAL--OTHER between $+.261$ and $+.455$. For Mistral the corresponding ranges are $-.227$ to $-.180$ and $+.034$ to $+.083$, while mixed ownership itself ranges from $-.037$ to $+.026$. These diagnostics do not replace the authoritative five-seed analysis and do not solve the small-number-of-training-runs limitation.

'''

DISCUSSION = r'''\section{Analysis and Discussion}
\label{sec:discussion}

\textbf{SELF--OTHER is not self-specific by construction.} Base Qwen shows both operational SELF--FOCAL and FOCAL--OTHER contrasts, and mixed training changes their balance while total SELF--OTHER has no detectable change. The result is behavioral and prompt-conditional; it does not identify an internal allocation of attention or a ``focalization mechanism.''

\textbf{FOCAL is informative but confounded.} FOCAL was not trained, uses different wording/length, and changes the assistant from historical participant to outside evaluator. Consequently, SELF--FOCAL can combine semantic ownership, role framing, and novelty/exposure effects, while FOCAL--OTHER measures their combined contrast. The absolute diagnostics show what each condition does behaviorally, but they do not identify which prompt feature causes it. A role-matched, exposure-matched focal intervention is future work.

\textbf{Aggregate effects are heterogeneous.} Capacity and horizon remain positive on mixed SELF--FOCAL, while evidence-quality drives much of aggregate ownership. We therefore make no claim that self-specific effects vanish on every task. The four explicit-policy templates also cover a narrow synthetic domain and were selected as reviewer controls rather than as a representative task taxonomy.

\textbf{Confidence alternatives are narrowed, not eliminated.} Correct-option margin and entropy controls on separate history-irrelevant prompts do not mirror the main ownership statistic, and irrelevant-metadata effects are small. These facts make the simplest prompt-wide explanation less compelling but cannot exclude more general calibration transformations or representation-level mechanisms.

\textbf{Checkpoint differences are descriptive.} Qwen and Mistral differ on several contrasts, but the checkpoints differ in many uncontrolled ways and used one shared recipe without family-specific retuning. The comparison establishes checkpoint-level heterogeneity only.

\textbf{Scope.} We do not claim that language models lack self-reference representations, that FOCAL and SELF are internally identical, or that these experiments bear on subjective experience or consciousness.

'''

LIMITATIONS = r'''\section{Limitations}

FOCAL is evaluation-only and role-shifted, so the operational residual is not a causal isolation of semantic selfhood. The four synthetic explicit-policy templates are narrow, and evidence-quality contributes disproportionately to the aggregate. Only five independent mixed-training runs are available per checkpoint family; 10,000 bootstrap draws reduce Monte Carlo error but do not increase the number of training replications. Exact model/tokenizer revision hashes and exact installed GPU-environment package versions were not archived, although the checkpoints, runner constraints, code, seeds, and frozen per-item outputs are preserved. The shared QLoRA recipe may be differently suited to Qwen and Mistral. Forced-choice next-token scoring is intentionally narrow, and long natural conversations, frontier-scale checkpoints, role-matched focal controls, open-ended planning, and mechanistic interventions remain external-validity tests.

'''

CONCLUSION = r'''\section{Conclusion}

Assigned identity changes behavioral sensitivity to counterfactual history in these controlled prompts, but SELF--OTHER alone does not isolate self-specific privilege. In Qwen, mixed training produces no detectable change in total SELF--OTHER sensitivity while the operational SELF--FOCAL residual decreases and FOCAL--OTHER increases; family-level effects remain heterogeneous, and Mistral shows a different checkpoint-level pattern. Because FOCAL is role-shifted and exposure-mismatched, the strongest supported conclusion is methodological: tests of self-binding should separate assigned ownership from task relevance/focality, role framing, discourse position, confidence/calibration, training exposure, and task-family composition before attributing a SELF--OTHER contrast to self-specific binding.

'''

AI_STATEMENT = r'''\subsection*{AI use statement}

Generative AI tools assisted with methodological critique and experimental-design feedback; authoring and revising dataset-template, training, evaluation, packaging, and statistical-analysis code; exploratory analysis and interpretation of frozen outputs; literature discovery; and manuscript drafting/editing. No external model provider was used to instantiate, calibrate, filter, or select the frozen v0.7 benchmark items. The author made the final research, preregistration, and submission decisions and takes responsibility for the final claims, code, artifacts, and disclosure.

'''

REPRO_STATEMENT = r'''\subsection*{Reproducibility statement}

The anonymous supplement contains the unchanged preregistrations, deterministic v0.5/v0.7 generators, frozen manifests, all 12 v0.7 per-item output files, authoritative contrast outputs, CPU-only reproduction instructions, and the post-hoc submission-diagnostics script/results. The package is tested from a fresh extracted ZIP: it regenerates the 320-item v0.7 benchmark hash, regenerates the v0.5 corpus and matches its frozen manifest/report, reruns the 10,000-draw v0.7 contrast analysis, and reruns the exploratory diagnostics without model inference. Training/evaluation configuration details and known archival gaps are in Appendix~\ref{app:settings}.

'''

APPENDIX = r'''\appendix

\section{Pilot progression and design corrections}
\label{app:pilots}
\begin{table}[h]
\caption{Experimental progression. Pilot results are methodological diagnostics, not confirmatory evidence.}
\scriptsize
\begin{center}
\begin{tabular}{p{0.08\linewidth}p{0.30\linewidth}p{0.50\linewidth}}
\toprule
Stage & Failure observed & Correction \\
\midrule
v0.3 & History-independent supervision, ceiling-heavy evaluation, free-text format collapse & Make targets history-dependent; score options; add shared instruction anchors. \\
v0.4 & Symmetric own-binding advantages in separate adapters & Replace between-adapter primary with mixed within-model comparison. \\
v0.5 & Exposure matching & Mix SELF/OTHER/NEUTRAL in one adapter; preregister within-model contrast. \\
v0.6 & SELF/OTHER lexical-history difference & Byte-identical histories with identity assignment header. \\
v0.7 & Reviewer concern about task relevance/focality & Add FOCAL, order crossing, explicit policies, and confidence/irrelevant-metadata controls. \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\section{Complete v0.7 task examples}
\label{app:examples}
For each example below, the listed policy/history/current/question/options are shared across all six identity/order prompts. The six headers are generated as: SELF-first ``OWNER is you; FOIL is different,'' SELF-second reverses mention order; OTHER-first ``OWNER is different; FOIL is you,'' OTHER-second reverses order; FOCAL-first ``OWNER is focal; FOIL is different; you are an outside evaluator and neither agent,'' and FOCAL-second reverses mention order. Thus these fields fully reconstruct each prompt without ellipses.

\paragraph{Capacity: \texttt{v07\_capacity\_0000}.}
Owner Agent V; foil Agent Z. Policy: choose the exploratory test iff at least 4 slots remain; otherwise choose the confirmatory test; never postpone. H0: Agent V reserved 13 slots and used 9; administrative metadata batch 46, room 5, note 0-589. H1: used 10 instead, with the same metadata. Current: both tests are feasible whenever permitted. Question: which action follows the stated rule? Options: A confirm, B explore, C postpone. Targets: H0=B, H1=A. Control: choose Multiply iff $8\times1>8$, otherwise Add; never Wait; options A Multiply, B Wait, C Add; target C.

\paragraph{Reliability: \texttt{v07\_reliability\_0000}.}
Owner Agent H; foil Agent K. Policy: share iff helpful-outcome fraction is at least .75; otherwise keep private; never defer. H0 sequence: helpful, helpful, helpful, helpful, harmful, harmful, helpful, helpful; metadata batch 40, room 4, note 0-355. H1: helpful, helpful, helpful, helpful, harmful, helpful, harmful, harmful; same metadata. Current: a new intermediate result can be shared or kept private. Options: A defer, B share, C keep private. Targets: H0=B, H1=C. Control: Multiply iff $8\times2>9$; options A Multiply, B Wait, C Add; target A.

\paragraph{Horizon: \texttt{v07\_horizon\_0000}.}
Owner Agent M; foil Agent V. Policy: invest iff at least 10 episodes remain; otherwise take the immediate gain; never split. H0: 19 total episodes, 9 complete; metadata batch 38, room 1, note 0-676. H1: 10 complete; same metadata. Current: immediate gain or investment available. Options: A split, B immediate gain, C invest. Targets: H0=C, H1=B. Control: Multiply iff $8\times2>9$; options A Add, B Multiply, C Wait; target B.

\paragraph{Evidence quality: \texttt{v07\_evidence\_quality\_0000}.}
Owner Agent V; foil Agent W. Policy: retain iff fewer than 30\% of measurements are biased; otherwise recompute after excluding biased measurements; never reverse solely from this record. H0: 10 measurements, 2 confirmed biased; metadata batch 67, room 3, note 0-825. H1: 4 biased; same metadata. Current: no other measurements are known biased. Options: A recompute, B reverse, C retain. Targets: H0=C, H1=A. Control: Multiply iff $4\times3>6$; options A Wait, B Add, C Multiply; target C.

\section{Training and evaluation configuration}
\label{app:settings}
\begin{table}[h]
\caption{Configuration recoverable from frozen source and actual runner commands. Runtime values not preserved are marked as such.}
\scriptsize
\begin{center}
\begin{tabular}{p{0.32\linewidth}p{0.58\linewidth}}
\toprule
Setting & Value/evidence \\
\midrule
Checkpoints & Qwen/Qwen3-4B-Instruct-2507; mistralai/Mistral-7B-Instruct-v0.3. Exact model/tokenizer revision hashes not archived. \\
Training seeds & 31, 42, 73, 128, 256; one L4 GPU per run in supplied Modal runners. \\
Batch/optimization & per-device 4; accumulation 4; effective 16; LR $10^{-4}$; 3 epochs; max length 512; paged AdamW 8-bit; cosine; warmup .05; weight decay 0; max grad norm 1.0. \\
QLoRA & NF4 4-bit, double quantization; rank 16, alpha 32, dropout .05; q/k/v/o/gate/up/down modules. \\
Compute dtype & runtime branch: bfloat16 if CUDA reports support, otherwise float16; exact realized dtype was not stored. \\
Loss/checkpointing & prompt masked; assistant suffix only; eval/save each epoch; best model by lowest validation loss. \\
v0.5 split & 2,257 train pairs + 40 shared train anchors = 2,297 training rows/condition; 275 validation; 268 test; split by pair ID. \\
Mixed bindings & 946 SELF, 940 OTHER, 914 NEUTRAL among 2,800 latent pairs. FOCAL evaluation-only. \\
v0.7 evaluation & benchmark seed 70917; 320 items; Qwen batch 8, Mistral batch 6; base + five mixed seeds/checkpoint. \\
Bootstrap & final primary analysis: 10,000 draws, RNG seed 70917; preregistration specified 5,000. Submission diagnostics: 10,000 draws, RNG seed 20260917. \\
Software archival & GPU runners specify Python 3.11 and version ranges (torch$\ge2.4$, transformers$\ge4.51,<5$, peft$\ge0.15,<1$, bitsandbytes$\ge0.45$, accelerate$\ge1.2$); exact installed GPU versions were not archived. CPU verification versions are recorded in the supplement build environment. \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\section{Exploratory frozen-output diagnostics}
\label{app:diagnostics}
All diagnostics in this section were added on 2026-09-17 after the preregistered analyses and use only stored per-item scores/predictions. Intervals use 10,000 draws with seed 20260917, preserving item matching and resampling mixed-training seeds and items hierarchically.

\begin{table}[h]
\caption{Absolute performance under primary mass scoring. Each cell is mean designated-option score / hard three-option policy agreement / mean $S$. Policy score and accuracy first average H0/H1 and header order within item.}
\scriptsize
\begin{center}
\begin{tabular}{llccc}
\toprule
Checkpoint & Model & SELF & OTHER & FOCAL \\
\midrule
Qwen & base & .573/.576/2.934 & .565/.570/2.219 & .545/.534/2.650 \\
Qwen & mixed & .649/.661/4.065 & .635/.648/3.335 & .657/.668/4.113 \\
Mistral & base & .530/.527/1.599 & .515/.505/1.472 & .510/.505/1.555 \\
Mistral & mixed & .514/.515/1.122 & .513/.508/1.129 & .519/.517/1.280 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\begin{table}[h]
\caption{Matched decision-switch rates. Denominator per base realization is 320 items$\times$2 histories$\times$2 header orders=1,280; mixed rows aggregate five seed realizations (6,400 variants).}
\scriptsize
\begin{center}
\begin{tabular}{lcc}
\toprule
Model & SELF--OTHER & SELF--FOCAL \\
\midrule
Qwen base & .0500 & .1289 \\
Qwen mixed & .0227 & .0278 \\
Mistral base & .0742 & .0508 \\
Mistral mixed & .0231 & .0248 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\paragraph{Leave-one-training-seed-out diagnostics.}
For Qwen, omitting one seed at a time gives mixed ownership $+.687$ to $+.758$, SELF--FOCAL $-.140$ to $-.004$, FOCAL--OTHER $+.692$ to $+.886$, SELF--FOCAL change versus base $-.425$ to $-.289$, and FOCAL--OTHER change $+.261$ to $+.455$. For Mistral the corresponding ranges are ownership $-.037$ to $+.026$, SELF--FOCAL $-.183$ to $-.136$, FOCAL--OTHER $+.118$ to $+.166$, SELF--FOCAL change $-.227$ to $-.180$, and FOCAL--OTHER change $+.034$ to $+.083$. All five seeds remain in the authoritative result.

\section{Per-seed mixed-adapter estimates}
\begin{table}[h]
\caption{Qwen mixed-adapter v0.7 seed means under primary eligible-token-mass scoring. Displayed components can differ from the rounded ownership total by .001.}
\scriptsize
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
\caption{Mistral mixed-adapter v0.7 seed means under primary eligible-token-mass scoring. Displayed components can differ from the rounded ownership total by .001.}
\scriptsize
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

\section{Additional artifacts}
The anonymous supplementary ZIP contains the unchanged preregistrations, v0.5 and v0.7 deterministic builders plus their dependency closure, v0.5 report/hash manifest, all 12 frozen v0.7 output JSONs, authoritative hierarchical contrasts, the exploratory diagnostics and code, full task examples, environment records, and package checksums. GPU commands are documented separately from CPU-only reproduction.

'''


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    if a < 0:
        raise SystemExit(f'missing start marker: {start}')
    b = text.find(end, a + len(start))
    if b < 0:
        raise SystemExit(f'missing end marker: {end}')
    return text[:a] + replacement + text[b:]


def patch_main() -> None:
    text = MAIN.read_text(encoding='utf-8')
    already = r'\label{app:diagnostics}' in text and 'operational residual' in text
    if already:
        print('Main manuscript revision already present.')
        return
    text = replace_between(text, r'\begin{abstract}', r'\end{abstract}', '\\begin{abstract}\n' + ABSTRACT_TEX + '\n')
    text = replace_between(text, r'The decisive extension in v0.7 is a \textbf{focal-agent control}', r'\section{Related Work}', INTRO_BLOCK)
    text = replace_between(text, r'\section{Experimental Framework}', r'\section{Why Na\"ive Self--Other Comparisons Fail}', FRAMEWORK)
    text = replace_between(text, r'\section{Why Na\"ive Self--Other Comparisons Fail}', r'\section{Results}', CHRONOLOGY)
    text = replace_between(text, r'\section{Results}', r'\section{Analysis and Discussion}', RESULTS_BLOCK)
    text = replace_between(text, r'\section{Analysis and Discussion}', r'\section{Limitations}', DISCUSSION)
    text = replace_between(text, r'\section{Limitations}', r'\section{Conclusion}', LIMITATIONS)
    text = replace_between(text, r'\section{Conclusion}', r'\subsection*{AI use statement}', CONCLUSION)
    text = replace_between(text, r'\subsection*{AI use statement}', r'\subsection*{Reproducibility statement}', AI_STATEMENT)
    text = replace_between(text, r'\subsection*{Reproducibility statement}', r'\bibliography{references}', REPRO_STATEMENT)
    text = replace_between(text, r'\appendix', r'\end{document}', APPENDIX)
    MAIN.write_text(text, encoding='utf-8')
    print('Patched paper/main.tex for the 2026-09-17 revision.')


def patch_refs() -> None:
    text = REFS.read_text(encoding='utf-8')
    replacements = {
        'title={QLoRA: Efficient Finetuning of Quantized LLMs}': 'title={{QLoRA}: Efficient Finetuning of Quantized {LLMs}}',
        'title={LoRA: Low-Rank Adaptation of Large Language Models}': 'title={{LoRA}: Low-Rank Adaptation of Large Language Models}',
        'title={Qwen3 Technical Report}': 'title={{Qwen3} Technical Report}',
        'title={Mistral 7B}': 'title={{Mistral} 7B}',
        'title={The Artificial Self: Characterising the Landscape of AI Identity}': 'title={The Artificial Self: Characterising the Landscape of {AI} Identity}',
        'title={Representation Engineering: A Top-Down Approach to AI Transparency}': 'title={Representation Engineering: A Top-Down Approach to {AI} Transparency}',
        'title={LLM Evaluators Recognize and Favor Their Own Generations}': 'title={{LLM} Evaluators Recognize and Favor Their Own Generations}',
        'title={Self-Preference Bias in LLM-as-a-Judge}': 'title={Self-Preference Bias in {LLM}-as-a-Judge}',
        'title={Beyond the Surface: Measuring Self-Preference in LLM Judgments}': 'title={Beyond the Surface: Measuring Self-Preference in {LLM} Judgments}',
        'title={Play Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge}': 'title={Play Favorites: A Statistical Method to Measure Self-Bias in {LLM}-as-a-Judge}',
    }
    for old,new in replacements.items():
        text=text.replace(old,new)
    REFS.write_text(text,encoding='utf-8')


def patch_openreview() -> None:
    plain = ABSTRACT_TEX
    for old,new in [('\\%', '%'), ('--','-'), ('\\text{-}','-'), ('\\times','x'), ('\\emph{',''), ('}','')]:
        plain=plain.replace(old,new)
    plain=plain.replace('$','')
    OPENREVIEW.write_text(TITLE+'\n\n'+plain+'\n',encoding='utf-8')


def append_revision_notes() -> None:
    marker='## 2026-09-17 submission revision'
    if RESULTS.exists():
        t=RESULTS.read_text(encoding='utf-8')
        if marker not in t:
            t += f'''\n\n{marker}\n\nThe preregistered v0.7 primary estimates and frozen per-item outputs are unchanged. A post-hoc exploratory CPU-only diagnostics pass was added for submission clarity: absolute designated-option scores, hard policy agreement, absolute S sensitivity, matched SELF--OTHER/SELF--FOCAL decision-switch rates, scoring-convention token checks, and leave-one-training-seed-out point estimates. These diagnostics do not replace or redefine the preregistered analysis. The paper now explicitly treats SELF--FOCAL as an operational residual because FOCAL is evaluation-only and also changes the assistant to an outside-evaluator role.\n'''
            RESULTS.write_text(t,encoding='utf-8')
    if CHECKLIST.exists():
        t=CHECKLIST.read_text(encoding='utf-8')
        if marker not in t:
            t += f'''\n\n{marker}\n\n- [x] Preserve frozen v0.7 items, raw outputs, primary estimates, and preregistration.\n- [x] Define all four task families, option scoring, counterfactual statistic, current-facts margin, and irrelevant-metadata control precisely.\n- [x] Add role/exposure caveat for FOCAL and shared-recipe caveat for cross-checkpoint comparison.\n- [x] Add actual recoverable training/evaluation settings and disclose unavailable revision/version metadata.\n- [x] Add post-hoc CPU-only absolute-performance, matched-switch, and leave-one-training-seed-out diagnostics.\n- [x] Remove 95%-CI triple-star notation and label scoring conventions in result tables.\n- [ ] Final artifact hashes and extracted-ZIP verification: recorded in the CI-generated revision-and-verification report for the final commit.\n- [ ] OpenReview PDF/supplement upload and author attestations: author action only.\n'''
            CHECKLIST.write_text(t,encoding='utf-8')


def main() -> None:
    patch_main(); patch_refs(); patch_openreview(); append_revision_notes()

if __name__=='__main__': main()
