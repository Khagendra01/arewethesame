# GPU handoff — assistant v0.3 pilot seed 31

This is the intentional boundary between dataset construction and model training.
Everything before this point runs without a model-provider API. The next step requires a CUDA GPU.

## Frozen data

The pilot is under:

```text
experiments/assistant_v03_pilot_seed31/
```

It contains 1,000 matched scene variants and 5,000 condition rows. Training files are already separated by condition and whole-life split:

```text
training/<neutral|self|other|shuffled_self|spp>/<train|validation|test>.jsonl
```

`sha256.json` is the frozen manifest. The trainer refuses to proceed if a selected file differs from the recorded hash. It also checks that all five conditions have identical pair order and identical target responses.

## Default first model

Use:

```text
Qwen/Qwen3-4B-Instruct-2507
```

The pilot uses 4-bit NF4 QLoRA and completion-only loss. Prompt tokens are masked with `-100`; only the assistant answer is supervised. This matters because the five conditions intentionally have different prompt lengths, while their matched target answers are identical.

Use exactly the same base model, seed, LoRA rank, learning rate, epochs, batch construction, and response supervision for every condition.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-train.txt
nvidia-smi
```

Recommended pilot GPU: an L4 24 GB or similar 24 GB CUDA GPU. The 4-bit 4B setup is intentionally conservative. A 16 GB GPU may work with a smaller per-device batch, but do not change effective batch size across conditions.

## First GPU smoke run

Run one condition first to catch CUDA/library issues:

```bash
python scripts/train_condition_lora.py --condition neutral
```

Expected input counts are 850 train examples and 50 validation examples. The 100 test examples are hash/alignment checked but are not used by `Trainer` for fitting or best-checkpoint selection.

The trainer writes:

```text
outputs/lora_pilot_seed31/neutral/
  run_metadata.json
  final_metrics.json
  best_adapter/
```

`outputs/` is gitignored.

## Train the matched five adapters

After the neutral smoke run succeeds, start from the same base model and run all five conditions with identical defaults:

```bash
bash scripts/train_all_conditions.sh
```

This produces:

```text
outputs/lora_pilot_seed31/
  neutral/
  self/
  other/
  shuffled_self/
  spp/
```

Do not tune hyperparameters separately by condition. If a global hyperparameter must change, rerun all five adapters.

## Experimental discipline

- Use validation loss for checkpoint selection; do not use the test split to choose hyperparameters.
- Keep seed 31 and the supplied whole-life split fixed for this pilot.
- Do not merge conditions into one training run; each adapter is a separate experimental treatment from the same base model.
- Do not compare adapters trained from different base-model revisions.
- Preserve `run_metadata.json` and `final_metrics.json` for every condition.
- After all five adapters are trained and the training recipe is frozen, run the locked behavioral evaluation and the held-out test split.

## What to bring back after GPU training

For each condition, keep the adapter directory and provide the small files below for analysis:

```text
run_metadata.json
final_metrics.json
trainer_state.json  # from the selected checkpoint, if present
```

The next analysis step is matched behavioral evaluation, not more dataset generation.
