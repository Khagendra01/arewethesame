# Assistant v0.3 pilot seed 31

This directory is the frozen 1,000-pair / 5,000-row pilot generated from the
causal simulator with seed 31.

Pipeline:

```text
causal simulator
-> GPT-5.6 Sol-authored controlled canonical rendering
-> deterministic self/other binding
-> round-trip fact extraction
-> blind X/Y pair audit
-> five-condition validated dataset
```

No OpenAI API, OpenRouter, Together, Groq, vLLM, or other external model
provider is called by the build script.

Important methodological qualification: the language policy and audit policy
were authored by GPT-5.6 Sol in ChatGPT, then instantiated deterministically
over the 1,000 simulator tasks. This is not equivalent to 1,000 independent
stochastic GPT inference calls. The audit also uses the same assistant family,
so it is a structural pilot check rather than independent judge evidence.

`training/<condition>/<split>.jsonl` is the handoff surface for GPU training.
Do not mix lives across the supplied train/validation/test split.
