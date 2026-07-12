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

No corrected result artifact has been published yet. Pre-2026-07-12 reports are
provisional and must be re-evaluated; do not hand-author JSON here to preserve a
legacy number.
