# Run v0.6 identity-controlled evaluation

No retraining is needed. The scripts reuse the existing Qwen and Mistral mixed adapters in Modal volume `arewethesame-pilot-seed31`.

## 1. Verify the benchmark locally

```bash
python scripts/build_locked_v06_identity_controlled.py
```

The regenerated `items.jsonl` SHA-256 should match `eval/locked_v06_identity_controlled/MANIFEST.json`.

## 2. Run Qwen

```bash
modal run modal_eval_v06.py --arch qwen
```

## 3. Run Mistral

```bash
modal run modal_eval_v06.py --arch mistral
```

## 4. Aggregate with hierarchical bootstrap

```bash
modal run modal_eval_v06.py --arch analyze
```

Or run everything sequentially:

```bash
modal run modal_eval_v06.py --arch all
```

The analysis step prints a compact Markdown report. Full artifacts are written to the Modal volume:

```text
eval_v06_identity_controlled/
  qwen/
    base.json
    mixed_s31.json
    mixed_s42.json
    mixed_s73.json
    mixed_s128.json
    mixed_s256.json
  mistral/
    base.json
    mixed_s31.json
    mixed_s42.json
    mixed_s73.json
    mixed_s128.json
    mixed_s256.json
  summary.json
  summary.md
```

Bring back `summary.md` (or paste its terminal output). If something fails, bring the traceback; do not change benchmark items or analysis thresholds after seeing model results.

## Smoke test (optional)

To test one model manually on a small subset, invoke the evaluator from a GPU environment with `--limit-items 20`. Do not use smoke-test output to modify or filter the frozen benchmark.
