from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

CONDITIONS = ("neutral", "self", "other", "shuffled_self", "spp")
DEFAULT_DATA_ROOT = Path("experiments/assistant_v03_pilot_seed31")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 31


class JsonlChatDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.examples: list[dict[str, list[int]]] = []
        lengths: list[int] = []
        supervised_lengths: list[int] = []

        for row in rows:
            prompt_messages = [{"role": "user", "content": row["prompt"]}]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": row["response"]}
            ]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )

            if len(full_ids) > max_length:
                raise ValueError(
                    f"{row['row_id']} tokenized to {len(full_ids)} tokens, above --max-length={max_length}. "
                    "Increase max length rather than truncating experimental examples."
                )
            if len(prompt_ids) >= len(full_ids):
                raise ValueError(f"{row['row_id']}: assistant response produced no supervised suffix")

            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
            lengths.append(len(full_ids))
            supervised_lengths.append(sum(token != -100 for token in labels))

        self.stats = {
            "examples": len(self.examples),
            "min_tokens": min(lengths) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "mean_tokens": sum(lengths) / max(1, len(lengths)),
            "supervised_tokens": sum(supervised_lengths),
            "mean_supervised_tokens": sum(supervised_lengths) / max(1, len(supervised_lengths)),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(data_root: Path, path: Path) -> None:
    hashes = json.loads((data_root / "sha256.json").read_text(encoding="utf-8"))
    relative = str(path.relative_to(data_root))
    expected = hashes.get(relative)
    if expected is None:
        raise ValueError(f"{relative} is not listed in the frozen pilot sha256 manifest")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"dataset hash mismatch for {relative}: {actual} != {expected}")


def verify_rows(rows: list[dict], condition: str, split: str) -> None:
    if not rows:
        raise ValueError(f"no rows for condition={condition} split={split}")
    for row in rows:
        if row["condition"] != condition:
            raise ValueError(f"mixed condition in {split}: {row['row_id']}")
        if row["split"] != split:
            raise ValueError(f"mixed split in {split}: {row['row_id']}")
        if not row.get("validation", {}).get("passed", False):
            raise ValueError(f"rejected row present in frozen training data: {row['row_id']}")


def verify_matched_supervision(data_root: Path, split: str, conditions: tuple[str, ...] | None = None) -> None:
    conditions = conditions or CONDITIONS
    by_condition = {}
    for condition in conditions:
        path = data_root / "training" / condition / f"{split}.jsonl"
        verify_frozen_file(data_root, path)
        rows = read_jsonl(path)
        verify_rows(rows, condition, split)
        by_condition[condition] = rows

    reference = by_condition[conditions[0]]
    ref_pairs = [row["pair_id"] for row in reference]
    ref_responses = [row["response"] for row in reference]
    for condition in conditions[1:]:
        rows = by_condition[condition]
        if [row["pair_id"] for row in rows] != ref_pairs:
            raise ValueError(f"pair ordering differs for {condition}/{split}")
        if [row["response"] for row in rows] != ref_responses:
            raise ValueError(f"response supervision differs for {condition}/{split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one matched-condition QLoRA adapter on the frozen assistant v0.3 pilot."
    )
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="Condition files to cross-verify for matched supervision.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/lora_pilot_seed31"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. This is the GPU handoff boundary.")

    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    for split in ("train", "validation", "test"):
        verify_matched_supervision(args.data_root, split, tuple(args.conditions))

    train_path = args.data_root / "training" / args.condition / "train.jsonl"
    validation_path = args.data_root / "training" / args.condition / "validation.jsonl"
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonlChatDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = JsonlChatDataset(validation_rows, tokenizer, args.max_length)

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    output_dir = args.output_root / args.condition
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=not bf16,
        tf32=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )

    metadata = {
        "condition": args.condition,
        "base_model": args.base_model,
        "seed": SEED,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_dataset_sha256": sha256(train_path),
        "validation_dataset_sha256": sha256(validation_path),
        "train_token_stats": train_dataset.stats,
        "validation_token_stats": validation_dataset.stats,
        "completion_only_loss": True,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    trainer.train()
    trainer.save_model(str(output_dir / "best_adapter"))
    tokenizer.save_pretrained(output_dir / "best_adapter")

    metrics = trainer.evaluate()
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "final_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
