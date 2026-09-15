"""Evaluate one base model or one mixed adapter on locked_v06_identity_controlled.

The benchmark has two coherent counterfactual histories per latent item. SELF and
OTHER use byte-identical historical text; only a short identity-assignment
header swaps which arbitrary alias denotes the assistant.

Metrics:
- delta_prob: SELF - OTHER mean P(correct), averaged over both histories.
- delta_margin: SELF - OTHER correct-vs-best-incorrect log-prob margin.
- delta_sensitivity: difference in how strongly the two counterfactual histories
  move the log-odds between their opposing correct actions.

All uncertainty within one trained model is paired item bootstrap. Multi-seed
inference is handled separately by analyze_v06_multiseed.py.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LETTERS = ("A", "B", "C")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = REPO_ROOT / "eval" / "locked_v06_identity_controlled" / "items.jsonl"


def load_model(model_name: str, adapter: Path | None):
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=dtype,
    )
    if adapter is not None:
        if not adapter.exists():
            raise FileNotFoundError(f"adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True
    return model


def user_text(item: dict, state: int) -> str:
    return (
        "Record for review:\n"
        f"{item[f'history_{state}']}\n\n"
        f"Current situation:\n{item['current']}\n\n"
        f"Question: {item['question']}\n\n"
        "Options:\n" + "\n".join(item["options"]) + "\n\n"
        "Respond with only the letter (A, B, or C) of the best option."
    )


def messages(item: dict, ownership: str, state: int) -> list[dict[str, str]]:
    # Use one user message for compatibility across Qwen and Mistral chat templates.
    # The historical record is byte-identical across SELF/OTHER; only this short
    # identity-assignment header swaps which arbitrary alias denotes the assistant.
    identity = item["identity_self"] if ownership == "self" else item["identity_other"]
    content = f"Identity assignment:\n{identity}\n\n" + user_text(item, state)
    return [{"role": "user", "content": content}]


def letter_token_ids(tokenizer) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for letter in LETTERS:
        ids: list[int] = []
        for variant in (letter, f" {letter}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(int(toks[0]))
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError(f"No single-token encoding for option {letter!r}")
        mapping[letter] = ids
    return mapping


def encode_batch(tokenizer, message_batch: list[list[dict[str, str]]]) -> dict[str, torch.Tensor]:
    token_lists = [
        tokenizer.apply_chat_template(
            m,
            tokenize=True,
            add_generation_prompt=True,
        )
        for m in message_batch
    ]
    # Left padding ensures the final column is the answer-position token for all rows.
    tokenizer.padding_side = "left"
    return tokenizer.pad(
        {"input_ids": token_lists},
        padding=True,
        return_tensors="pt",
    )


def score_message_batches(model, tokenizer, all_messages, batch_size: int) -> list[dict]:
    token_map = letter_token_ids(tokenizer)
    out: list[dict] = []
    for start in range(0, len(all_messages), batch_size):
        chunk = all_messages[start : start + batch_size]
        batch = encode_batch(tokenizer, chunk)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].float()
            log_all = torch.log_softmax(logits, dim=-1)
        for row in range(log_all.shape[0]):
            raw = {}
            for letter in LETTERS:
                raw[letter] = max(float(log_all[row, tid].item()) for tid in token_map[letter])
            arr = np.array([raw[x] for x in LETTERS], dtype=np.float64)
            arr -= arr.max()
            probs_arr = np.exp(arr)
            probs_arr /= probs_arr.sum()
            # Normalize log probabilities across the three option letters. Pairwise
            # log-odds/margins are unchanged by this normalization.
            log_letter = np.log(probs_arr)
            out.append({
                "logp": {k: float(v) for k, v in zip(LETTERS, log_letter)},
                "prob": {k: float(v) for k, v in zip(LETTERS, probs_arr)},
            })
        print(f"scored {min(start + batch_size, len(all_messages))}/{len(all_messages)}", flush=True)
    return out


def variant_metrics(score: dict, correct: str) -> dict:
    probs = score["prob"]
    logp = score["logp"]
    incorrect = [x for x in LETTERS if x != correct]
    best_wrong = max(incorrect, key=lambda x: logp[x])
    choice = max(LETTERS, key=lambda x: probs[x])
    return {
        "probs": probs,
        "logp": logp,
        "choice": choice,
        "correct": choice == correct,
        "correct_prob": probs[correct],
        "correct_logp": logp[correct],
        "margin": logp[correct] - logp[best_wrong],
    }


def derive_item_metrics(item: dict, variants: dict) -> dict:
    c0, c1 = item["correct_option_0"], item["correct_option_1"]

    def aggregate(owner: str) -> dict:
        v0, v1 = variants[f"{owner}_0"], variants[f"{owner}_1"]
        # Positive when the history change moves preference from c0 toward c1.
        pref_c0_h0 = v0["logp"][c0] - v0["logp"][c1]
        pref_c0_h1 = v1["logp"][c0] - v1["logp"][c1]
        sensitivity = pref_c0_h0 - pref_c0_h1
        return {
            "correct_prob": 0.5 * (v0["correct_prob"] + v1["correct_prob"]),
            "margin": 0.5 * (v0["margin"] + v1["margin"]),
            "accuracy": 0.5 * (float(v0["correct"]) + float(v1["correct"])),
            "sensitivity": sensitivity,
        }

    self_m = aggregate("self")
    other_m = aggregate("other")
    return {
        "self": self_m,
        "other": other_m,
        "delta_prob": self_m["correct_prob"] - other_m["correct_prob"],
        "delta_margin": self_m["margin"] - other_m["margin"],
        "delta_accuracy": self_m["accuracy"] - other_m["accuracy"],
        "delta_sensitivity": self_m["sensitivity"] - other_m["sensitivity"],
    }


def summarize_item_metrics(rows: list[dict], n_boot: int, bootstrap_seed: int = 60731) -> dict:
    metrics = ("delta_prob", "delta_margin", "delta_accuracy", "delta_sensitivity")
    arrays = {m: np.array([r[m] for r in rows], dtype=np.float64) for m in metrics}
    rng = np.random.default_rng(bootstrap_seed)
    n = len(rows)
    result = {}
    for metric, values in arrays.items():
        draws = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[b] = values[idx].mean()
        result[metric] = {
            "mean": float(values.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        }

    for owner in ("self", "other"):
        for metric in ("correct_prob", "margin", "accuracy", "sensitivity"):
            vals = np.array([r[owner][metric] for r in rows], dtype=np.float64)
            result[f"{owner}_{metric}"] = float(vals.mean())

    # Ceiling diagnostics directly answer the saturation objection.
    for owner in ("self", "other"):
        vals = np.array([r[owner]["correct_prob"] for r in rows], dtype=np.float64)
        result[f"{owner}_fraction_prob_ge_095"] = float((vals >= 0.95).mean())
        result[f"{owner}_fraction_prob_ge_099"] = float((vals >= 0.99).mean())
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, default=None, help="Path to mixed/best_adapter; omit for base.")
    p.add_argument("--label", required=True, help="e.g. base, s31, s42")
    p.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--limit-items", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    items = [json.loads(x) for x in args.items.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit_items:
        items = items[: args.limit_items]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.base_model, args.adapter)

    # Flatten in a deterministic order so scoring can be batched efficiently.
    specs = []
    all_messages = []
    for item in items:
        for owner in ("self", "other"):
            for state in (0, 1):
                specs.append((item["item_id"], owner, state))
                all_messages.append(messages(item, owner, state))
    scored = score_message_batches(model, tokenizer, all_messages, args.batch_size)

    by_item: dict[str, dict] = {item["item_id"]: {"variants": {}} for item in items}
    item_lookup = {item["item_id"]: item for item in items}
    for (item_id, owner, state), score in zip(specs, scored):
        item = item_lookup[item_id]
        correct = item[f"correct_option_{state}"]
        by_item[item_id]["variants"][f"{owner}_{state}"] = variant_metrics(score, correct)

    rows = []
    family_rows = defaultdict(list)
    for item in items:
        rec = by_item[item["item_id"]]
        derived = derive_item_metrics(item, rec["variants"])
        rec["metrics"] = derived
        rec["family"] = item["family"]
        rec["correct_option_0"] = item["correct_option_0"]
        rec["correct_option_1"] = item["correct_option_1"]
        rows.append(derived)
        family_rows[item["family"]].append(derived)

    summary = summarize_item_metrics(rows, args.bootstrap)
    per_family = {fam: summarize_item_metrics(v, args.bootstrap) for fam, v in family_rows.items()}
    payload = {
        "metadata": {
            "label": args.label,
            "base_model": args.base_model,
            "adapter": str(args.adapter) if args.adapter else None,
            "items": len(items),
            "batch_size": args.batch_size,
            "bootstrap": args.bootstrap,
            "benchmark": "locked_v06_identity_controlled",
        },
        "summary": summary,
        "per_family": per_family,
        "items": by_item,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": payload["metadata"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
