# Recover the v0.6 evidence package from Modal

The original v0.6 inference outputs were written to the persistent Modal volume `arewethesame-pilot-seed31` under the volume-relative directory `eval_v06_identity_controlled` (mounted as `/outputs/eval_v06_identity_controlled` during the run).

## 1. Download the original outputs

From the repository root:

```bash
mkdir -p evidence/v06
modal volume get arewethesame-pilot-seed31 eval_v06_identity_controlled evidence/v06
```

Expected files after download (depending on Modal CLI directory nesting, the `eval_v06_identity_controlled/` directory may appear directly inside `evidence/v06`; point `--root` below at the directory that contains `qwen/` and `mistral/`):

```text
qwen/base.json
qwen/mixed_s31.json
qwen/mixed_s42.json
qwen/mixed_s73.json
qwen/mixed_s128.json
qwen/mixed_s256.json
mistral/base.json
mistral/mixed_s31.json
mistral/mixed_s42.json
mistral/mixed_s73.json
mistral/mixed_s128.json
mistral/mixed_s256.json
summary.json
summary.md
```

Do not regenerate inference outputs or replace the frozen v0.6 benchmark. These are the original run artifacts.

## 2. Run the reviewer-targeted reanalysis

If `evidence/v06` directly contains `qwen/` and `mistral/`:

```bash
python scripts/analyze_v06_reviewer_checks.py \
  --root evidence/v06 \
  --output evidence/v06/reviewer_checks.json \
  --bootstrap 5000
```

The script intentionally fails if any of the twelve expected result files lacks exactly 840 item IDs, if a run has missing SELF/OTHER x H0/H1 variants, or if Qwen/Mistral item IDs differ. It does not silently analyze an intersection of incomplete result sets.

It reports:

- base and mixed `Delta_sensitivity` for both checkpoint families;
- paired post-training-minus-base contrasts;
- a direct Qwen-minus-Mistral interaction with seed resampling independent across families and shared item resampling;
- per-family estimates;
- leave-one-seed-out sensitivity;
- ownership-induced argmax decision-switch rates;
- mean absolute SELF/OTHER score changes;
- per-variant score distributions, entropy, and near-zero/near-one saturation diagnostics.

## 3. Preserve the original artifacts

Before editing or filtering anything, copy the downloaded directory unchanged into the anonymous supplementary artifact. Record:

- benchmark SHA-256: `153e2af5c2e5578ae54094e8d7d43ca8536b5976a27901a56f086b42ba581aac`;
- evaluated code commit/branch;
- base-model identifiers;
- adapter paths/seeds;
- `reviewer_checks.json` generated above.

Any alternate scoring analysis or later reviewer-driven benchmark must be stored separately from these original v0.6 outputs.
