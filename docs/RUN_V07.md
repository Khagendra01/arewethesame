# Run v0.7 reviewer controls

This is a **post-v0.6 reviewer-driven extension**. It reuses existing base models and v0.5 mixed adapters; no training is performed.

## Verify the frozen benchmark locally

```bash
git fetch origin
git checkout v07/reviewer-controls
rm -rf /tmp/v07_verify
python scripts/build_locked_v07_reviewer_controls.py --out /tmp/v07_verify --per-family 80
sha256sum /tmp/v07_verify/items.jsonl
```

Expected SHA-256:

```text
d4960832076326ba5ed31ff3664095dea8c916ea3fabf7e482bcde8207f5cc7a
```

Do not run GPU evaluation if the hash differs.

## Run all base + mixed evaluations

```bash
modal run modal_eval_v07.py --arch all
```

Equivalent separate calls:

```bash
modal run modal_eval_v07.py --arch qwen
modal run modal_eval_v07.py --arch mistral
modal run modal_eval_v07.py --arch analyze
```

Outputs are stored on the `arewethesame-pilot-seed31` Modal volume under:

```text
/outputs/eval_v07_reviewer_controls/
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
```

## Primary outputs to return

The `summary.json` contains both scoring conventions. The preregistered primary convention is `conventions.mass`, which sums eligible single-token option mass before A/B/C normalization. `conventions.max` is the legacy v0.6 scoring robustness check.

Please return at least:

- Qwen mixed `delta_ownership_sensitivity` under `mass`;
- Mistral mixed `delta_ownership_sensitivity` under `mass`;
- direct Qwen-minus-Mistral interaction;
- Qwen/Mistral `ownership_by_order_interaction`;
- `delta_self_vs_focal_sensitivity`;
- `delta_control_margin` and `delta_control_entropy`;
- `delta_irrelevant_sensitivity`;
- the corresponding primary values under legacy `max` scoring.

Interpretation rules are frozen in `docs/PREREGISTRATION_V07_REVIEWER_CONTROLS.md`.
