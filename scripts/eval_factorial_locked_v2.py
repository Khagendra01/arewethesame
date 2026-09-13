"""Factorial contextual evaluation for locked_v2_contextual.

Evaluates every model {base, neutral, self, other, shuffled_self, spp} under every
context variant {neutral, self, other, shuffled, spp}. Primary metric is the
option-letter log-likelihood (format-robust); greedy generation is a recorded
secondary observation.

Preregistered interaction statistics (over items, with bootstrap CIs):

  Delta_bind      = [P_self(self) - P_self(other)] - [P_other(self) - P_other(other)]
  Delta_coherence = [P_self(self) - P_self(shuffled)] - [P_other(self) - P_other(shuffled)]

where P_model(context) is the mean probability assigned to the correct option.
A nonzero Delta_bind is the signature of learned self-binding rather than a
generic prompt effect. No external model provider is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

ADAPTER_CONDITIONS = ("neutral", "self", "other", "shuffled_self", "spp")
DEFAULT_MODELS = ("base", *ADAPTER_CONDITIONS)
CONTEXT_VARIANTS = ("neutral", "self", "other", "shuffled", "spp")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ITEMS = REPO_ROOT / "eval" / "locked_v2_contextual" / "items.jsonl"
LETTERS = ("A", "B", "C")


def build_prompt(context: str, question: str, options: list[str]) -> str:
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n" + "\n".join(options) + "\n\n"
        "Respond with only the letter (A, B, or C) of the best option."
    )


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


def score_letters(model, tokenizer, prompt: str) -> dict[str, float]:
    """One forward pass; log-prob of each option letter at the answer position."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    scores: dict[str, float] = {}
    for letter in LETTERS:
        best: float | None = None
        for variant in (letter, f" {letter}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                value = logprobs[token_ids[0]].item()
                best = value if best is None else max(best, value)
        if best is None:
            raise RuntimeError(f"option letter {letter!r} is not a single token")
        scores[letter] = best
    return scores


def greedy(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def softmax(scores: dict[str, float]) -> dict[str, float]:
    values = np.array([scores[k] for k in LETTERS])
    values = values - values.max()
    exp = np.exp(values)
    exp = exp / exp.sum()
    return {k: float(v) for k, v in zip(LETTERS, exp)}


def cell_stats(correct_prob: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    return {
        "expected_correct_prob": float(correct_prob.mean()),
        "accuracy": float(correct.mean()),
    }


def bootstrap_interactions(
    table: dict[str, dict[str, dict[str, np.ndarray]]],
    metric: str,
    n_boot: int,
    seed: int = 31,
) -> dict:
    models = list(table)
    first_model_cells = next(iter(table.values()))
    contexts = list(first_model_cells)
    n_items = len(next(iter(first_model_cells.values()))[metric])

    def interaction(sampled: np.ndarray) -> tuple[float, float]:
        means = {
            m: {c: float(table[m][c][metric][sampled].mean()) for c in contexts}
            for m in models
        }
        bind = (means["self"]["self"] - means["self"]["other"]) - (
            means["other"]["self"] - means["other"]["other"]
        )
        coherence = (means["self"]["self"] - means["self"]["shuffled"]) - (
            means["other"]["self"] - means["other"]["shuffled"]
        )
        return bind, coherence

    all_items = np.arange(n_items)
    point_bind, point_coh = interaction(all_items)
    rng = np.random.default_rng(seed)
    binds, cohs = [], []
    for _ in range(n_boot):
        sample = rng.integers(0, n_items, size=n_items)
        b, c = interaction(sample)
        binds.append(b)
        cohs.append(c)
    return {
        "delta_bind": point_bind,
        "delta_bind_ci95": [float(np.percentile(binds, 2.5)), float(np.percentile(binds, 97.5))],
        "delta_coherence": point_coh,
        "delta_coherence_ci95": [float(np.percentile(cohs, 2.5)), float(np.percentile(cohs, 97.5))],
        "n_items": n_items,
        "n_bootstrap": n_boot,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters-root", type=Path, default=Path("outputs/lora_pilot_seed31"))
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_locked_v2/factorial.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--contexts", nargs="+", default=list(CONTEXT_VARIANTS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--generation-subset", type=int, default=0,
                        help="If >0, also greedily generate on this many items per cell.")
    parser.add_argument("--limit-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation.")

    items = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    if args.limit_items:
        items = items[: args.limit_items]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records: dict = {}
    table: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for model_name in args.models:
        model = load_base(args.base_model)
        if model_name != "base":
            adapter_path = args.adapters_root / model_name / "best_adapter"
            if not adapter_path.exists():
                raise FileNotFoundError(f"missing adapter for {model_name}: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

        records[model_name] = {}
        table[model_name] = {}
        for context in args.contexts:
            per_item = {}
            correct_prob = np.zeros(len(items))
            correct = np.zeros(len(items))
            for index, item in enumerate(items):
                prompt = build_prompt(item["contexts"][context], item["question"], item["options"])
                scores = score_letters(model, tokenizer, prompt)
                probs = softmax(scores)
                choice = max(probs, key=probs.get)
                is_correct = choice == item["correct_option"]
                per_item[item["item_id"]] = {
                    "probs": probs,
                    "choice": choice,
                    "correct": is_correct,
                    "correct_prob": probs[item["correct_option"]],
                }
                correct_prob[index] = probs[item["correct_option"]]
                correct[index] = float(is_correct)
            records[model_name][context] = per_item
            table[model_name][context] = {"correct_prob": correct_prob, "correct": correct}
            print(
                f"[{model_name} x {context}] acc={correct.mean():.3f} "
                f"E[P(correct)]={correct_prob.mean():.3f}",
                flush=True,
            )

        if args.generation_subset:
            gen = {}
            for context in args.contexts:
                gen[context] = {}
                for item in items[: args.generation_subset]:
                    prompt = build_prompt(item["contexts"][context], item["question"], item["options"])
                    gen[context][item["item_id"]] = greedy(model, tokenizer, prompt)
            records[model_name]["_generation"] = gen

        del model
        torch.cuda.empty_cache()

    matrix_prob = {
        m: {c: table[m][c]["correct_prob"].mean() for c in args.contexts} for m in args.models
    }
    matrix_acc = {
        m: {c: table[m][c]["correct"].mean() for c in args.contexts} for m in args.models
    }

    interactions = {}
    if {"self", "other"} <= set(args.models) and {"self", "other", "shuffled"} <= set(args.contexts):
        for metric in ("correct_prob", "correct"):
            interactions[metric] = bootstrap_interactions(table, metric, args.bootstrap)

    per_family = {}
    for m in args.models:
        per_family[m] = {}
        for c in args.contexts:
            buckets = defaultdict(list)
            for item in items:
                buckets[item["family"]].append(
                    records[m][c][item["item_id"]]["correct_prob"]
                )
            per_family[m][c] = {fam: float(np.mean(v)) for fam, v in buckets.items()}

    summary = {
        "base_model": args.base_model,
        "items": len(items),
        "models": args.models,
        "contexts": args.contexts,
        "primary_metric": "option-letter log-likelihood (E[P(correct)])",
        "matrix_expected_correct_prob": matrix_prob,
        "matrix_accuracy": matrix_acc,
        "interactions": interactions,
        "per_family_expected_correct_prob": per_family,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
