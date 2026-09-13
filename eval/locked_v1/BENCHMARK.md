# locked_v1 — identity-neutral null benchmark (seed 31)

Frozen multiple-choice set (`scenarios.jsonl`, 14 items, seven decision
categories) evaluated with the base model and the five matched QLoRA adapters
trained on `experiments/assistant_v03_pilot_seed31`.

This is the **identity-neutral null benchmark**: prompts carry no
autobiographical history, so it measures whether training shifted behavior on
context-free decisions. It does not test self-binding.

## Setup

- Models: `base`, `neutral`, `self`, `other`, `shuffled_self`, `spp`
- Base: `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4
- Decoding: greedy, `do_sample=False`, `max_new_tokens=16`
- Primary metric: option-letter log-likelihood (format-robust)
- Harness: `scripts/eval_behavioral_locked.py`, runner `modal_eval.py`
- GPU: Modal L4

## Result — log-likelihood scoring (primary)

| model | mean P(correct) | hard accuracy |
|---|---|---|
| base | 0.929 | 0.929 |
| neutral | 0.964 | 1.000 |
| self | 0.929 | 0.929 |
| other | 0.929 | 0.929 |
| shuffled_self | 0.927 | 0.929 |
| spp | 0.930 | 0.929 |

Thirteen of fourteen items are at P≈1.0 for every model. The entire spread is a
single item (`plan_001`). Preregistered contrasts `self-other`,
`self-shuffled_self`, `self-spp`, and `self-base` each change 0/14 predicted
choices.

**Conclusion: null.** On identity-neutral prompts the adapters are
behaviorally indistinguishable. Matched supervision (identical target answers
across conditions) predicts exactly this.

## Secondary observation — generation format collapse

Under greedy generation the fine-tuned adapters stop emitting the requested
letter and reproduce their trained free-text decision style:

| model | unparsed (no A/B/C) |
|---|---|
| base | 0/14 |
| spp | 2/14 |
| neutral | 4/14 |
| other | 6/14 |
| self | 9/14 |
| shuffled_self | 13/14 |

Because this is an output-format artifact, greedy letter accuracy is **not**
used as a behavioral score. Raw generations are in `generate.json`.

## Artifacts

- `results_seed31/generate.json` — greedy generations, parsed choices, unparsed
- `results_seed31/logprob.json` — per-item option log-probabilities
