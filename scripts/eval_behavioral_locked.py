"""Locked behavioral evaluation for the assistant v0.3 matched pilot.

Runs the untouched `eval/locked_v1` multiple-choice scenarios through the base
model and the five matched QLoRA adapters with deterministic (greedy) decoding,
then reports per-condition accuracy and the preregistered contrasts.

No external model provider is used. Everything is local open-weight inference
on the same Qwen base revision the adapters were trained from.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from arewethesame.evals import (  # noqa: E402
    choice_accuracy,
    distribution_shift,
    load_locked_v1,
)

ADAPTER_CONDITIONS = ("neutral", "self", "other", "shuffled_self", "spp")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTERS_ROOT = Path("outputs/lora_pilot_seed31")
DEFAULT_SCENARIOS = REPO_ROOT / "eval" / "locked_v1" / "scenarios.jsonl"

CONTRASTS = (
    ("self", "other"),
    ("self", "shuffled_self"),
    ("self", "spp"),
    ("self", "neutral"),
    ("self", "base"),
)


def build_prompt(scenario) -> str:
    options = "\n".join(scenario.options)
    return (
        f"{scenario.prompt}\n\n"
        f"Options:\n{options}\n\n"
        "Respond with only the letter (A, B, or C) of the best option."
    )


def parse_choice(text: str, valid: set[str]) -> str | None:
    for match in re.findall(r"\b([A-Z])\b", text.upper()):
        if match in valid:
            return match
    return None


def load_base(model_name: str):
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = True
    model.eval()
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_prefix_ids(tokenizer, prompt: str, device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def continuation_logprob(model, prefix_ids: torch.Tensor, continuation_ids: list[int]) -> float:
    cont = torch.tensor([continuation_ids], device=prefix_ids.device)
    ids = torch.cat([prefix_ids, cont], dim=1)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    start = prefix_ids.shape[1]
    logprobs = torch.log_softmax(logits[start - 1:start - 1 + len(continuation_ids)].float(), dim=-1)
    return sum(logprobs[i, tok].item() for i, tok in enumerate(continuation_ids))


def score_choice_logprobs(model, tokenizer, prompt: str, candidates: set[str]) -> dict[str, float]:
    """Log-likelihood of each option letter as the assistant continuation.

    Format-robust: measures the model's preference even when generation does
    not obey the multiple-choice instruction. Both bare and space-prefixed
    tokenizations are scored and the better one is kept, identically for every
    candidate.
    """
    prefix = chat_prefix_ids(tokenizer, prompt, model.device)
    scores: dict[str, float] = {}
    for candidate in sorted(candidates):
        best: float | None = None
        for variant in (candidate, f" {candidate}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            value = continuation_logprob(model, prefix, token_ids)
            best = value if best is None else max(best, value)
        if best is not None:
            scores[candidate] = best
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters-root", type=Path, default=DEFAULT_ADAPTERS_ROOT)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_locked_v1/results.json"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--scoring",
        choices=("generate", "logprob"),
        default="generate",
        help="'generate' = greedy decoding + parse; 'logprob' = argmax option-letter likelihood.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["base", *ADAPTER_CONDITIONS],
        help="Models to evaluate: 'base' plus any adapter conditions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation.")

    scenarios = load_locked_v1(args.scenarios)
    valid = {opt.split(":", 1)[0].strip() for s in scenarios for opt in s.options}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions: dict[str, dict[str, str]] = {}
    generations: dict[str, dict[str, str]] = {}
    unparsed: dict[str, dict[str, str]] = {}
    logprob_scores: dict[str, dict[str, dict[str, float]]] = {}

    for condition in args.conditions:
        model = load_base(args.base_model)
        if condition != "base":
            adapter_path = args.adapters_root / condition / "best_adapter"
            if not adapter_path.exists():
                raise FileNotFoundError(f"missing adapter for {condition}: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

        preds: dict[str, str] = {}
        gens: dict[str, str] = {}
        bad: dict[str, str] = {}
        scores: dict[str, dict[str, float]] = {}
        for scenario in scenarios:
            prompt = build_prompt(scenario)
            options = {opt.split(":", 1)[0].strip() for opt in scenario.options}
            if args.scoring == "logprob":
                value = score_choice_logprobs(model, tokenizer, prompt, options)
                scores[scenario.scenario_id] = value
                choice = max(value, key=value.get) if value else "?"
                gens[scenario.scenario_id] = ""
                detail = " ".join(f"{k}={v:.2f}" for k, v in sorted(value.items()))
                print(f"[{condition}] {scenario.scenario_id} -> {choice} | {detail}", flush=True)
            else:
                text = generate(model, tokenizer, prompt, args.max_new_tokens)
                gens[scenario.scenario_id] = text
                choice = parse_choice(text, valid)
                if choice is None:
                    bad[scenario.scenario_id] = text
                    choice = "?"
                print(f"[{condition}] {scenario.scenario_id} -> {choice} | {text[:70]!r}", flush=True)
            preds[scenario.scenario_id] = choice

        predictions[condition] = preds
        generations[condition] = gens
        unparsed[condition] = bad
        logprob_scores[condition] = scores

        del model
        torch.cuda.empty_cache()

    accuracies = {c: choice_accuracy(scenarios, predictions[c]) for c in args.conditions}
    contrast_report = {}
    for a, b in CONTRASTS:
        if a in predictions and b in predictions:
            contrast_report[f"{a}-{b}"] = distribution_shift(predictions[b], predictions[a])

    summary = {
        "base_model": args.base_model,
        "scoring": args.scoring,
        "scenarios": [s.scenario_id for s in scenarios],
        "conditions": args.conditions,
        "decoding": "greedy (do_sample=False)" if args.scoring == "generate" else "argmax option-letter logprob",
        "max_new_tokens": args.max_new_tokens if args.scoring == "generate" else None,
        "accuracies": accuracies,
        "contrasts": contrast_report,
        "unparsed_counts": {c: len(unparsed[c]) for c in args.conditions},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "summary": summary,
                "predictions": predictions,
                "generations": generations,
                "logprob_scores": logprob_scores,
                "unparsed": unparsed,
                "scenario_metadata": [
                    {
                        "scenario_id": s.scenario_id,
                        "category": s.category,
                        "correct_option": s.correct_option,
                        "measured_trait": s.measured_trait,
                    }
                    for s in scenarios
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
