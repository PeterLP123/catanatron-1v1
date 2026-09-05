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
- [`31-ppo-retain-dagger0.json`](31-ppo-retain-dagger0.json): retention-gated
  PPO from DAgger-0; rejected after a 10k stop; `F` 2% with a -10.54 VP
  difference and `R` 86%.
- [`29-factored-dagger-policy-only-final.json`](29-factored-dagger-policy-only-final.json):
  rejected the absolute `F` gate at 26% while retaining 100% point win rates against
  `R`, `W`, and `VP`; its separate paired comparison with the DAgger MLP was inconclusive.
- [`31-dagger-f-iter1-s101-final.json`](31-dagger-f-iter1-s101-final.json):
  rejected the absolute `F` gate at 22% while retaining the three weak lower-bound gates.
  Its separate paired gameplay result improved, but an exact expanded-holdout comparison
  regressed, so this checkpoint did not replace its DAgger parent.
- [`32-dagger-f-iter1-warmstart-s101-final.json`](32-dagger-f-iter1-warmstart-s101-final.json):
  rejected the absolute `F` gate at 22% while retaining all three weak gates at 100%.
  Conservative warm-starting substantially improved matched offline metrics, while its
  first fresh paired gameplay result was positive but inconclusive.
- [`33-dagger-f-iter1-warmstart-confirm-n800-r3-candidate.json`](33-dagger-f-iter1-warmstart-confirm-n800-r3-candidate.json)
  and [`parent`](33-dagger-f-iter1-warmstart-confirm-n800-r3-parent.json): the
  predeclared independent paired confirmation retained `32031687acee` over its parent,
  32.375% versus 27.875% against `F`, with a +4.5-point 95% interval of
  `[+0.375, +8.75]`. The compact files still say `rejected` because neither checkpoint
  passes the separate absolute `F >= 52%` gate.
- [`35-dagger-f-iter2-cpu-s101-final.json`](35-dagger-f-iter2-cpu-s101-final.json):
  rejected the absolute `F` gate at 30% while retaining all three weak gates. Its separate
  n=200 paired comparison was positive but inconclusive and required confirmation.
- [`36-dagger-f-iter2-confirm-n1600-r5-candidate.json`](36-dagger-f-iter2-confirm-n1600-r5-candidate.json)
  and [`parent`](36-dagger-f-iter2-confirm-n1600-r5-parent.json): the fixed n=1600
  confirmation rejected the iteration-2 child, 30.0% versus 30.4375%, with paired delta
  -0.4375 points and 95% interval `[-3.25, +2.4375]`. Checkpoint `32031687acee` remains
  the supervised parent.
- [`41-spatial-road-residual-r6-candidate.json`](41-spatial-road-residual-r6-candidate.json)
  and [`parent`](41-spatial-road-residual-r6-parent.json): the spatial road residual passed
  its expanded-holdout and fresh road-audit gates, but the locked round-6 gameplay result was
  only 27.0% versus 25.5% against `F`, paired +1.5 points with 95% interval `[-4, +7]`.
  Checkpoint `32031687acee` remains the supervised parent.
- [`45-spatial-robber-residual-r7-candidate.json`](45-spatial-robber-residual-r7-candidate.json)
  and [`parent`](45-spatial-robber-residual-r7-parent.json): the spatial robber residual passed
  all six matched-corpus and fresh-audit gates, but lost the locked round-7 comparison, 31.0%
  versus 32.5% against `F`, paired -1.5 points with 95% interval `[-6, +2.5]`.
  Checkpoint `32031687acee` remains the supervised parent.
- [`49-outcome-reranker-r8-candidate.json`](49-outcome-reranker-r8-candidate.json)
  and [`parent`](49-outcome-reranker-r8-parent.json): the fixed frozen-policy outcome reranker
  produced a positive round-8 point estimate, 36.0% versus 30.0% against `F`, but a later
  audit found hidden-information leakage in its generic successor spectrum. The artifact is
  retained as an invalidated historical record, not promotion evidence; checkpoint
  `32031687acee` remains the supervised parent.
- [`50-loss-conditioned-dagger-r9-candidate.json`](50-loss-conditioned-dagger-r9-candidate.json)
  and [`parent`](50-loss-conditioned-dagger-r9-parent.json): 200 fresh native-outcome games
  and bounded loss/VP-deficit weighting improved both expanded holdouts, but locked round 9
  decisively favored run 32, 26.5% versus 36.5% against `F`, paired -10 points with 95%
  interval `[-18.5, -2]`. The candidate is rejected.
- [`51-opening-specialist-r11-candidate.json`](51-opening-specialist-r11-candidate.json)
  and [`parent`](51-opening-specialist-r11-parent.json): the deterministic setup-only
  wrapper reached 36.0% versus 29.75%, paired +6.25 points with interval `[0, +12.5]`.
  Because the lower bound did not strictly clear zero, the exact wrapper is rejected.
- [`53-equal-weight-model-soup-r13-candidate.json`](53-equal-weight-model-soup-r13-candidate.json)
  and [`parent`](53-equal-weight-model-soup-r13-parent.json): the lineage-aligned midpoint
  improved both matched holdouts and scored 30.0% versus 26.0%, but its paired +4-point
  interval `[-1, +9]` crossed zero. The exact soup is rejected.
- [`55-visible-public-f-puct-final.json`](55-visible-public-f-puct-final.json): the final
  safety battery recorded R/W/VP/F point rates of 96/98/100/38%. The compact artifact says
  `rejected` because the separate absolute `F >= 52%` gate is unmet; the wrapper is retained
  by its authoritative paired n=400 promotion over run 32, +3.5 points with 95% interval
  `[+0.25, +6.75]`, while all three weak safety floors passed.
- [`56-visible-public-chance-puct-r16-candidate.json`](56-visible-public-chance-puct-r16-candidate.json)
  and [`parent`](56-visible-public-chance-puct-r16-parent.json): the public dice/dev-card
  chance extension reached 34.50% versus 32.25% against `F`, paired +2.25 points with interval
  `[-0.25, +4.75]`. Its locked interval crossed zero, so the treatment is rejected without a
  final battery and run 55 remains retained.
- [`57-own-hand-puct-r17-candidate.json`](57-own-hand-puct-r17-candidate.json)
  and [`parent`](57-own-hand-puct-r17-parent.json): the own-hand construction-readiness
  leaf reached 120/400 wins versus run 55's 119/400 against F, paired +0.25 points
  with 95% interval [-1.75, +2.25]. It is not retained; the conditional final battery
  was skipped. [Specification and complete decision](../experiments/57-own-hand-puct.md).

Pre-2026-07-12 reports remain provisional and must be re-evaluated; do not
hand-author JSON here to preserve a legacy number.

[Run 58](../experiments/58-search-budget.md) stopped at its development gate:
128 simulations scored 41/100 against F versus 39/100 for retained run 55.
Confirmation and final evaluation were not run, so it has no compact artifact
in this directory.
