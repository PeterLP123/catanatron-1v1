# Tracked model evidence

This directory stores compact promotion and final-evaluation summaries, not
weights or full run directories. Publish one with:

```bash
python examples/colonist_1v1_publish_result.py \
  runs/<run>/final_benchmark.json \
  --output docs/results/<experiment>.json
```

The publisher rejects development-seed reports, missing games, evaluator
errors, missing per-game evidence, one-seat evaluations, non-canonical or
mutated gates, forged aggregates, and reports without a checkpoint hash. Each
artifact records the source-report hash, checkpoint hash, protocol, outcome,
aggregate matchup results, and a hash of the omitted per-game rows. A rejected
model can still be published: failure is evidence, while bad accounting is not.

Published corrected evidence:

- [`20-hard-bc-actual-s101.json`](20-hard-bc-actual-s101.json): rejected final
  benchmark after 500k PPO; no lower-bound gates passed and `F` remained at 0%.
- [`22-hybrid-bc-raw-f-final.json`](22-hybrid-bc-raw-f-final.json): rejected
  final-seed `F` gate, but the raw hybrid-BC checkpoint won 24% with a -2.50 VP
  difference before PPO.
- [`26-hybrid-bc-parent-promotion.json`](26-hybrid-bc-parent-promotion.json):
  reconstructed hybrid-BC control on the promotion suite; `F` 20% with a -3.18
  VP difference; R/W/VP held.
- [`28-dagger-f-s101.json`](28-dagger-f-s101.json): first DAgger F iteration,
  kept over that parent; `F` 36% with a -1.80 VP difference; R/W/VP 100%.
- [`29-dagger-f-iter1.json`](29-dagger-f-iter1.json): second DAgger F iteration,
  discarded; `F` 34% with a -2.18 VP difference, worse than iteration 0.

Pre-2026-07-12 reports remain provisional and must be re-evaluated; do not
hand-author JSON here to preserve a legacy number.
