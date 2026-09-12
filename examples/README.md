# Pilot example

Generate a small deterministic pilot:

```bash
arewethesame --seed 7 --lives 2 --episodes 6 \
  --out outputs/pilot.jsonl \
  --report outputs/bias_report.json
```

This produces 12 latent events and five matched renderings per event (`neutral`, `self`, `other`, `shuffled_self`, `spp`) for 60 total rows.

The current leakage report compares `self` vs `other` for lexical similarity, length ratio, emotional-word imbalance, and motivation/mortality-word imbalance. These are sanity checks, not a complete bias metric.
