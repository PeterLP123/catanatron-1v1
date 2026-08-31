# Results log

This file separates historical observations from promotion-grade evidence.

> **Evidence reset — 2026-07-12.** Every evaluation in the provisional historical
> section predates the repaired game-accounting path. The old accumulator could omit turn-limit games from the
> denominator and final-VP averages. Reports also predate reliable fixed-seat
> scheduling, so their `seat0/seat1` fields are invalid. All old numbers are
> **provisional historical estimates**, not current scorecards, promotion evidence,
> or proof that any checkpoint is the best or most seat-balanced model.

Corrected and diagnostic results now establish a useful failure boundary: the hybrid-BC
checkpoint has material raw strength against `F`, the subsequent 500k PPO run removes it,
and two forward-KL anchor sweeps do not retain it past the first 10k-step check. One bounded
`F`-labelled DAgger iteration then produced a statistically supported improvement over that
hybrid-BC parent, although the child still loses most games to `F` and does not pass the
absolute strength gate.
A capacity-matched factored architecture subsequently improved held-out regret but produced
only an inconclusive +1-point paired gameplay delta, so it did not replace the DAgger MLP.
A second DAgger corpus then produced a scratch-trained child with a positive paired gameplay
interval, but exact re-evaluation of both checkpoints on the expanded holdout found worse
validation and test regret. The new data is preserved; the regressed checkpoint is not.
A conservative retrain from the retained parent then improved expanded-holdout regret by
about 13% and produced another positive gameplay point estimate, but its initial n=200
paired interval crossed zero. Its predeclared independent n=800 confirmation then cleared
zero, so the conservative child is retained over its parent. It still loses most games to
`F` and is not an absolutely strong bot. A third 100-game corpus improved offline metrics,
but its predeclared n=1600 confirmation was neutral to slightly negative. Generic hard-state
weighting then selected the unchanged parent at epoch 0. These failures establish a bounded
uniform-imitation plateau rather than a reason to keep adding identically labelled data. A
topology-aware road residual then improved both the expanded holdout and a fresh audit-only
road set, but its round-6 gameplay gain was only +1.5 points with a 95% interval crossing
zero. A matched fixed-search road-label pilot then failed its deeper-search stability,
agreement, and latency gates, so no search-labelled treatment was trained. Finally, a narrow
robber residual improved both matched and untouched-audit metrics, yet lost its locked
round-7 gameplay comparison. Better imitation of the same `F` labels is therefore not enough
to promote these action-family treatments. A separately trained outcome critic subsequently
beat public-score baselines on all eight held-out gates. Its one fixed frozen-policy reranker
gave a positive +6-point gameplay estimate, but a later audit found hidden-information
leakage in its successor spectrum. The wrapper run is invalidated; the critic's separate
offline evidence remains useful, but the composite is not promoted.
Three later, deliberately different treatments did not displace run 32. A setup-only value
wrapper and an equal-weight parent/child model soup both improved their locked gameplay point
estimates, but their paired intervals touched or crossed zero. A directly trained shared
action-conditioned head failed before gameplay because both matched holdout regrets and legal
accuracy regressed substantially. A hidden-information-safe same-turn PUCT then separated the
search mechanism from its leaf value. The outcome-critic version missed its development gate;
the public-F version cleared a fresh 400-game paired promotion interval and retained the weak
battery. Run 55 is therefore the retained playable wrapper, while run 32 remains its frozen
supervised policy and the absolute `F >= 52%` target remains unmet. A public-chance extension
then passed its operational and development gates, but its locked promotion interval crossed
zero; run 56 is rejected without tuning and does not displace run 55.

A clean repository-local reconstruction independently preserved the earlier boundary:
promotion-suite `F` moved from 20% for the hybrid-BC parent to 36% for DAgger iteration 0,
then fell to 34% on iteration 1; retention-gated PPO from iteration 0 collapsed to 2% at its
10k stop. Those local artifacts support reproducibility but do not supersede the later locked
paired decisions above. No candidate has passed the absolute 52% `F` gate.

## Evidence standard for new entries

A comparable result must include:

- evaluation schema `1.1`, Colonist 1v1 rules, an explicit protocol, and both seats;
- a `promotion` or `final` seed suite separate from development selection;
- requested games equal accounted games, with zero evaluator errors;
- one per-game outcome for every request, including truncations and final VP;
- the checkpoint SHA-256, Git commit, protocol game count, and gate mode;
- confidence-lower-bound gates for promotion/final claims.

Compact accepted or rejected summaries should be published under [`docs/results/`](results/README.md).
Development reports are useful for iteration but must not be copied into this ledger as final evidence.

## Provisional historical observations

These numbers are retained so hypotheses and artifact lineage are not lost. Exact win rates,
VP margins, weighted scores, and every seat split must be replaced by a corrected rebaseline.

| Method | Artifact | Legacy result vs `F` | Legacy R / W / VP | Evidence status |
|---|---|---:|---:|---|
| Self-play PPO, 500k | `runs/ec2_proxy_500k` | 0.5% | 93 / 89 / 83 | Provisional; missing-game and seat-accounting risk |
| BC, 5.4M rows | `runs/v2/bc.pt` | 0.5% | 81 / 55 / 67 | Provisional; legacy row split/objective and old evaluator |
| PPO trained against `F`, 500k | `runs/v2_ppo_fheavy` | about 1% | 97 / 85 / 86 | Provisional; no matching corrected final report |
| One-ply search with learned value | `runs/v3/value.pt` | 0.5% | 37 / not run / not run | Provisional; old evaluator |

### Legacy `ec2_proxy_500k` report

- **Date:** 2026-06-28
- **Eval commit:** `0c00e81` plus uncommitted evaluation changes
- **Legacy report:** `runs/ec2_proxy_500k/eval_two_seat_no_m200.json`
- **Claim boundary:** aggregate figures may indicate a large weakness against `F` and
  `AB:2`, but the exact rates and all seat interpretations are invalid until re-run.

| Opponent | Legacy gate | Legacy win rate | Legacy VP diff | Status now |
|---|---:|---:|---:|---|
| `R` | 90% | 92.9% | +9.91 | Provisional |
| `W` | 70% | 88.5% | +8.77 | Provisional |
| `VP` | 60% | 83.3% | +7.49 | Provisional |
| `F` | 52% | 0.5% | -10.80 | Provisional |
| `AB:2` | 52% | 0.0% | -10.76 | Provisional |
| `G:25` | 52% | Not run | Not run | Missing |
| `M:200` | 52% | Not run | Not run | Missing |

The legacy weighted score was reported as `0.396` with three of five local point gates.
It must not be compared with the repaired uncertainty-aware weighted score.

### Legacy `v2_ppo_fheavy` report

The following old table is kept only to identify the checkpoint that needs re-evaluation.
The earlier description of this model as the “best available” or “most seat-balanced” policy
has been withdrawn because the reports do not support that ranking.

| Opponent | Legacy win rate | Legacy VP diff | Status now |
|---|---:|---:|---|
| `R` | 97.4% | +10.40 | Provisional |
| `W` | 85.3% | +7.90 | Provisional |
| `VP` | 85.6% | +7.86 | Provisional |
| `F` | 1.0% | -10.10 | Provisional |
| `AB:2` | 0.5% | -10.50 | Provisional |

## Working hypotheses, not findings

The historical runs motivate tests but do not prove a root cause:

- corrected final evidence confirms a material gap against `F`: experiment 20 finished
  at 0/50 wins and -11.76 average VP difference;
- the hybrid legal-CE plus listwise checkpoint reached 12/50 wins and -2.50 VP difference
  against `F` before PPO, so PPO forgetting is now the primary tested hypothesis;
- the full hybrid parent retained `R=100%`, `W=98%`, and `VP=100%`, so its `F=24%` result
  is not explained by a general collapse against the weak battery;
- anchor coefficients from `0` through `10` all hit the 10k retention stop; coefficients
  `3` and `10` reached only 1/20 against `F`, while every smaller coefficient was 0/20;
- the completed `05-mcts-strength-sweep` found only a small search signal at 100 ms
  (5-10% wins against `F`) with p95 latency around 283 ms, so `F` remains the stronger
  available teacher for DAgger;
- one 100-game DAgger F iteration from the reconstructed hybrid-BC parent improved
  promotion-suite `F` from 20% / -3.18 VP to 36% / -1.80 VP and is kept;
- a second 100-game iteration from that student fell to `F` 34% / -2.18 VP and is
  discarded; stop this DAgger line rather than growing the corpus;
- retention-gated PPO from that kept parent (`coef=10`) hit the 10k stop at
  development `F=5%` and promotion `F=2%` / `-10.54` VP; discard the PPO child;
- the completed DAgger child beat `F` in 67/200 matched promotion-schedule games versus
  40/200 for its parent: paired score delta +0.135, 95% bootstrap CI `[+0.06, +0.21]`,
  with VP difference improving from -3.815 to -2.390;
- the capacity-matched factored policy-only model lowered validation/test mean regret to
  0.07724/0.07588 from the DAgger MLP's 0.08248/0.08183, but on a fresh 200-game paired
  schedule it won 56 games versus 54 for the MLP: +0.010 with 95% CI `[-0.065, +0.085]`;
- adding a 0.25 win-value auxiliary loss improved test win-value accuracy from 62.5% to
  77.9% but worsened test policy regret from 0.07588 to 0.08391, so policy-only was selected;
- the scratch-trained second DAgger child beat its parent on a fresh matched schedule,
  59/200 versus 44/200 against `F` (+7.5 points, 95% CI `[+0.49, +15]`, VP gain +0.395),
  but matched expanded validation/test regret worsened from 0.08229/0.08175 to
  0.08474/0.08456, so the two-part retention gate rejected the checkpoint;
- initializing from the retained parent and admitting it as epoch 0 lowered matched
  validation/test regret to 0.07147/0.07116 and raised legal-choice accuracy by
  3.11/2.96 points; fresh round-2 gameplay was 36.5% versus 32.0% for the parent
  (+4.5 points, 95% CI `[-4, +13.5]`, VP gain +0.555), so gameplay remained inconclusive;
- the independently seeded round-3 confirmation reproduced the +4.5-point gain on n=800:
  259 wins (32.375%) versus 223 (27.875%), 95% paired bootstrap CI
  `[+0.375, +8.75]` points, with average VP difference improving by +0.64875;
- iteration 2 improved matched validation/test regret by 0.00170/0.00188 and its first
  n=200 point estimate favored the child by 3 points, but the predeclared n=1600 round-5
  confirmation favored the retained parent: 30.0% versus 30.4375%, paired -0.4375 points,
  95% CI `[-3.25, +2.4375]`, and VP-difference delta -0.248125;
- generic hard-state weighting regressed at every trained epoch, so epoch 0 was retained and
  the offline gate stopped the run before a paired schedule was allocated;
- the old factored policy is worse than the retained MLP on the 70-shard expanded holdout:
  candidate-minus-parent regret +0.00557 validation/+0.00455 test and about -2.04 points
  legal-choice accuracy on both splits;
- `BUILD_ROAD` is the largest residual regret contributor on the exact combined 70-shard
  holdout; this is a diagnostic that motivates a fresh road-specific audit, not evidence
  that an untested architecture or teacher will improve gameplay;
- the frozen run-40 audit contains 9,522 decisions from 100 alternating-seat parent-visited
  games, 10 verified shards, and zero truncations; it was never used for training or epoch
  selection;
- the run-41 spatial residual improved expanded validation/test regret by 0.00371/0.00347.
  On the untouched audit halves, road total regret fell 147.40 to 139.23 and 166.08 to
  151.83 without lowering road accuracy or worsening aggregate regret. Fresh round-6 gameplay
  was nevertheless only 27.0% versus 25.5%, paired +1.5 points with 95% interval
  `[-4, +7]`, so checkpoint `d7b106bd603c` is rejected;
- the run-43 matched road-label pilot rejected M800 as a teacher: round agreement was 72.09%,
  exact agreement conditional on both choosing roads was 50%, p95 label latency exceeded
  2.05 s, and M200 agreed with M800 only 67.44%; no training corpus was authorized;
- the frozen run-44 audit contains 889 `MOVE_ROBBER` choices from 100 alternating-seat games,
  10 verified shards, and zero truncations; it was never used for training, epoch selection,
  or hyperparameter tuning;
- the run-45 robber residual improved matched validation/test regret by
  0.002427/0.001939 and fresh-audit regret by 0.003170/0.001290 without lowering audit
  accuracy. Locked round-7 gameplay still favored the parent, 31.0% versus 32.5%, paired
  -1.5 points with 95% interval `[-6, +2.5]`, so checkpoint `dc58347a6642` is rejected;
- the run-46 audit found 100% win-target and 95.43% VP-margin row coverage across 622,939
  rows, with 4,300 trajectories in 2,300 whole-game split groups and no cross-logical-corpus
  collisions; its public-score baselines were win AUC 0.84543/Brier 0.16443 and margin
  MAE 4.47225/RMSE 5.71370;
- the run-47 value-only critic passed all eight validation/test gates. On test it improved
  win AUC to 0.88535 and Brier to 0.13731, while margin MAE/RMSE fell to 2.63152/3.67852;
- the fixed run-48 top-3/0.05 reranker appeared to pass its six operational gates, but a
  2026-08-31 audit found that its generic chance spectrum exposed hidden robber/Monopoly
  outcomes. Runs 48/49 are invalidated; their 36.0% versus 30.0% point estimate and paired
  +6-point interval `[-2, +14.5]` are retained only as historical, non-promotional records;
- run 50 added 200 fresh run-32-visited games and bounded loss/VP-deficit weights. It improved
  expanded validation/test regret by 0.00671/0.00589 and legal-choice accuracy by 0.55/0.72
  points, yet locked round-9 gameplay regressed decisively: 26.5% versus 36.5% for run 32,
  paired -10 points with 95% interval `[-18.5, -2]`. This treatment is rejected;
- old per-seat differences cannot support any seat-balance conclusion;
- full AlphaZero-style training is a gated fallback, not the established next solution.

## Corrected evidence ledger

| Date | Experiment | Checkpoint hash | Seed suite | Result | Artifact |
|---|---|---|---|---|---|
| - | Corrected historical rebaseline | - | `final` | Not run | - |
| 2026-07-15 | `05-mcts-strength-sweep` | N/A | Held-out search seeds | Complete diagnostic; 100 ms search reached 5-10% vs `F` | Run artifact only |
| - | Legal-CE BC baseline | - | `promotion` / `final` | Not run | - |
| - | Listwise BC treatment | - | `promotion` / `final` | Not run | - |
| 2026-08-30 | `26-hybrid-bc-parent-promotion` | `c63e7ca21f85` | `promotion` | Reconstructed hybrid-BC control; F 20% / -3.18 VP; R/W/VP 98/100/96%; F gate rejected | [`26-hybrid-bc-parent-promotion.json`](results/26-hybrid-bc-parent-promotion.json) |
| 2026-08-30 | `28-dagger-f-s101` | `d5fab233652b` | `promotion` | DAgger iteration 0 kept; F 36% / -1.80 VP; R/W/VP 100/100/100%; F gate rejected | [`28-dagger-f-s101.json`](results/28-dagger-f-s101.json) |
| 2026-08-30 | `29-dagger-f-iter1` | `99146ca2b06a` | `promotion` | DAgger iteration 1 discarded; F 34% / -2.18 VP; R/W/VP 100/100/100%; worse than iteration 0 | [`29-dagger-f-iter1.json`](results/29-dagger-f-iter1.json) |
| 2026-08-30 | `31-ppo-retain-dagger0` | `7aec88f93822` | `promotion` | Anchored PPO from DAgger-0 rejected; 10k retention stop; F 2% / -10.54 VP; R 86% failed | [`31-ppo-retain-dagger0.json`](results/31-ppo-retain-dagger0.json) |
| 2026-08-10 | `28-dagger-f-s101` paired validation | `b366433c5c7` | explicit promotion schedule | Accepted over parent; 33.5% vs 20.0% `F`, paired +13.5 points (95% CI +6 to +21), VP diff +1.425 | UCL `paired-promotion-f-n200-20260810/paired_comparison.json` (`56a922c9c1d4`) |
| 2026-08-10 | `29-factored-dagger-s101` | `6dc7528ed909` | `final` | Not promoted; 28% vs 27% `F`, paired +1 point (95% CI -6.5 to +8.5), absolute R/W/VP/F 100/100/100/26% | [`29-factored-dagger-policy-only-final.json`](results/29-factored-dagger-policy-only-final.json); UCL paired artifact `591108859a66` |
| 2026-08-10 | `31-dagger-f-iter1-s101` | `ed9b87946bb2` | `promotion` / `final` round 1 | Checkpoint rejected; paired `F` gain +7.5 points, but matched validation/test regret both regressed; absolute R/W/VP/F 100/98/94/22% | [`31-dagger-f-iter1-s101-final.json`](results/31-dagger-f-iter1-s101-final.json); UCL paired artifact `bb1601e6aebd` |
| 2026-08-10 | `32-dagger-f-iter1-warmstart-s101` | `32031687acee` | `promotion` / `final` round 2 | Offline gate passed strongly; paired `F` point gain +4.5 but 95% CI crossed zero; absolute R/W/VP/F 100/100/100/22% | [`32-dagger-f-iter1-warmstart-s101-final.json`](results/32-dagger-f-iter1-warmstart-s101-final.json); UCL paired artifact `0fdf40e79842` |
| 2026-08-10 | `33-dagger-f-iter1-warmstart-confirm-n800-r3` | `32031687acee` | `promotion` round 3 | Retained over parent; 32.375% vs 27.875% `F`, paired +4.5 points (95% CI +0.375 to +8.75), VP diff +0.64875; still fails absolute `F` gate | [`candidate`](results/33-dagger-f-iter1-warmstart-confirm-n800-r3-candidate.json) / [`parent`](results/33-dagger-f-iter1-warmstart-confirm-n800-r3-parent.json); paired artifact `f7476b1316e1` |
| 2026-08-10 | `35-dagger-f-iter2-cpu-s101` | `7ec5729d2d47` | `promotion` / `final` round 4 | Offline gate passed; paired n=200 favored child by 3 points but CI crossed zero; absolute R/W/VP/F 100/100/100/30% | [`35-dagger-f-iter2-cpu-s101-final.json`](results/35-dagger-f-iter2-cpu-s101-final.json); paired artifact `8b56b1d0` |
| 2026-08-10 | `36-dagger-f-iter2-confirm-n1600-r5` | `7ec5729d2d47` | `promotion` round 5 | Rejected; child 30.0% vs retained parent 30.4375%, paired -0.4375 points (95% CI -3.25 to +2.4375), VP diff -0.248125 | [`candidate`](results/36-dagger-f-iter2-confirm-n1600-r5-candidate.json) / [`parent`](results/36-dagger-f-iter2-confirm-n1600-r5-parent.json); paired artifact `e26f26771c00` |
| 2026-08-10 | `38-dagger-f-hard-states-cpu-s101` | epoch-0 fallback | matched 70-shard holdout | Rejected offline; every trained epoch regressed, selected checkpoint exactly matched parent, paired/final evaluation skipped | Run artifact `matched-holdout.json` (`6fa57668f565`) |
| 2026-08-10 | `41-spatial-road-residual-cpu-s101` | `d7b106bd603c` | `promotion` round 6 | Rejected; all six offline/audit gates passed, but candidate 27.0% vs parent 25.5% `F`, paired +1.5 points (95% CI -4 to +7), VP diff +0.095 | [`candidate`](results/41-spatial-road-residual-r6-candidate.json) / [`parent`](results/41-spatial-road-residual-r6-parent.json); paired report `fb6d19b0abab` |
| 2026-08-10 | `42-road-teacher-calibration-s20260802` | N/A | matched training-state diagnostic | Mechanism gate passed on 98 rows; authorized only the larger teacher pilot, not training | Local run artifacts; stability `407c94deff97`, novelty `83acf18de0dc` |
| 2026-08-10 | `43-road-teacher-quality-s20260803` | N/A | matched training-state diagnostic | Rejected teacher branch; M800 failed agreement, conditional road agreement, latency, and M200-proxy gates | Local run artifacts; M800 stability `91efbfecd92f`, M800-vs-F `701d6eac97fc` |
| 2026-08-10 | `44-robber-audit-s20260804` | retained parent | fresh audit-only holdout | Frozen before treatment work; 889 MOVE_ROBBER rows, 10 shards, zero truncations; prohibited from training/tuning | Local immutable manifest `063135546993` |
| 2026-08-10 | `45-spatial-robber-residual-cpu-s101` | `dc58347a6642` | `promotion` round 7 | Rejected; all six offline/audit gates passed, but candidate 31.0% vs parent 32.5% `F`, paired -1.5 points (95% CI -6 to +2.5), VP diff -0.065 | [`candidate`](results/45-spatial-robber-residual-r7-candidate.json) / [`parent`](results/45-spatial-robber-residual-r7-parent.json); paired report `705031922b5a` |
| 2026-08-10 | `46-outcome-target-audit-s101` | N/A | frozen whole-game audit | Accepted as an input gate; 100% win and 95.43% margin coverage, 2,300 split groups, all eight coverage/integrity gates passed | Local audit report `5c6172e657e8` |
| 2026-08-10 | `47-factored-outcome-critic-cpu-s101` | `bb3cf7012fe8` | frozen validation/test splits | Accepted offline; value-only critic beat public-VP AUC, Brier, margin MAE, and RMSE baselines on both splits (8/8 gates) | Local offline report `9f2af557be33` |
| 2026-08-10 | `48-outcome-reranker-operational-s20260805` | manifest `d4f06bc623a3` | non-promotional diagnostic | **Invalidated 2026-08-31:** generic chance spectrum crossed the hidden-information boundary; historical 6/6 operational gates only | Local diagnostic `6e54bf8c87fe` |
| 2026-08-10 | `49-outcome-reranker-promotion-r8` | manifest `d4f06bc623a3` | `promotion` round 8 | **Invalidated 2026-08-31:** 36.0% vs 30.0% `F` and paired +6 points (95% CI -2 to +14.5) are not valid promotion evidence | [`candidate`](results/49-outcome-reranker-r8-candidate.json) / [`parent`](results/49-outcome-reranker-r8-parent.json); paired report `2478e7c3fd17` |
| 2026-08-30 | `50-loss-conditioned-dagger-s20260830` | `f44e95993476` | `promotion` round 9 | Rejected; offline regret improved strongly, but candidate 26.5% vs parent 36.5% `F`, paired -10 points (95% CI -18.5 to -2), VP diff -0.74 | [`candidate`](results/50-loss-conditioned-dagger-r9-candidate.json) / [`parent`](results/50-loss-conditioned-dagger-r9-parent.json); paired report `9dd061b3fc54` |
| 2026-08-30 | `51-opening-specialist-s20260830` | manifest `d385ad1e1afe` | `promotion` round 11 | Rejected narrowly; setup-only wrapper 36.0% vs parent 29.75% `F`, paired +6.25 points (95% CI 0 to +12.5), VP diff +1.295; strict lower bound did not clear zero | [`candidate`](results/51-opening-specialist-r11-candidate.json) / [`parent`](results/51-opening-specialist-r11-parent.json); paired report `a4084994ca08` |
| 2026-08-30 | `52-action-conditioned-policy-s101` | `6dd9ce2a4ef3` | matched 60-shard holdout | Rejected offline; validation/test regret worsened by +0.01738/+0.01660 and legal accuracy fell 5.47/5.53 points, so gameplay was skipped | Run artifact `matched-holdout.json` (`610fa730d895`) |
| 2026-08-30 | `53-equal-weight-model-soup-s101` | `06edf98678f6` | `promotion` round 13 | Rejected; midpoint improved both holdouts, then scored 30.0% vs parent 26.0% `F`, paired +4 points (95% CI -1 to +9), VP diff +0.1625 | [`candidate`](results/53-equal-weight-model-soup-r13-candidate.json) / [`parent`](results/53-equal-weight-model-soup-r13-parent.json); paired report `362f04dbd646` |
| 2026-08-30 | `54-visible-same-turn-puct-s20260830` | manifest `1c398828cc4d` | `dev` round 54 | Rejected before promotion; critic-guided search passed all 9 operational gates, then improved `F` by only +2 points (35% vs 33%) against the required +3, despite VP diff +0.28 | Local paired report `79e0147f4e34`; promotion round 14 untouched |
| 2026-08-30 | `55-visible-public-f-puct-s20260830` | manifest `36e4707e7621` | `promotion` / `final` round 15 | Retained playable wrapper; 35.75% vs 32.25% `F`, paired +3.5 points (95% CI +0.25 to +6.75), VP diff +0.525; final R/W/VP/F 96/98/100/38% | [`55-visible-public-f-puct-final.json`](results/55-visible-public-f-puct-final.json); paired report `ed18d345c2eb` |
| 2026-08-30 | `56-visible-public-chance-puct-s20260830` | manifest `7d22c13caa7f` | `promotion` round 16 | Rejected; public dice/dev-card chance search passed 12/12 operational gates and dev, then scored 34.50% vs 32.25% `F`, paired +2.25 points (95% CI -0.25 to +4.75), VP diff +0.3525; final skipped | [`candidate`](results/56-visible-public-chance-puct-r16-candidate.json) / [`parent`](results/56-visible-public-chance-puct-r16-parent.json); paired report `1cbdb86b3081` |
| 2026-07-16 | `20-hard-bc-actual-s101` | `2f4ab72a895a` | `final` | Rejected; 0/4 gates, `F` 0%, weighted score 0.3315 | [`20-hard-bc-actual-s101.json`](results/20-hard-bc-actual-s101.json) |
| 2026-07-17 | `22-hybrid-bc-raw-f-final` | `886f5b374011` | `final` | Rejected F gate; 24% wins, -2.50 VP difference | [`22-hybrid-bc-raw-f-final.json`](results/22-hybrid-bc-raw-f-final.json) |
| 2026-07-17 | `23-hybrid-bc-full-final` | `886f5b374011` | `final` | Rejected F gate; R/W/VP 100/98/100%, F 24%, weighted score 0.5733 | UCL run artifact |
| 2026-07-17 | `24-bc-anchor-sweep` | multiple | diagnostic `final` | Rejected; coefficients 0/0.01/0.03/0.10 stopped at 10k, F 0/20 each | UCL run artifacts |
| 2026-07-17 | `25-bc-anchor-scale` | multiple | diagnostic `final` | Rejected; coefficients 0.3/1/3/10 stopped at 10k, best F 1/20 | UCL run artifacts |
| 2026-07-17 | `27-teacher-population-screen` | N/A | held-out diagnostic | Complete 28/28 two-game cells; no teacher promoted from n=2 evidence | UCL run artifact |

The `28` result retains the DAgger iteration as a better supervised checkpoint; it does not
pass the absolute `F >= 52%` strength gate. Both sides used seed `22260707`, 100 games in each
seat, 200/200 accounted games, and zero evaluator errors. Candidate and baseline checkpoint
SHA-256 values are `b366433c5c7f80810a154395b4a5016a93f1f253c3a370e783850029d48b375e`
and `886f5b374011e400a42b28c62dc0c3cd5de93b6175636efdcc651bcae4cdb464`.

The `29` policy-only checkpoint used 691,710 parameters versus 747,852 for the MLP control.
Both paired sides completed the same 200 unique seat/seed schedules with zero errors or
truncations. Candidate and control averaged -3.04 and -3.22 VP against `F`; the paired VP
difference gain was +0.18. The separate 50-game-per-opponent final battery accounted for all
200 games and retained all three weak gates, but `F=13/50` had a Wilson lower bound of 15.9%.
The paired artifact SHA-256 is
`591108859a66cfe7654d010fd9c59cb8a2d67727858e8233419ca93b1ed919f7`.

The `31` run added 100 alternating-seat games in immutable iteration 1: 9,494 rows,
zero truncations, iteration-manifest SHA-256
`96ac236d4f262c4b70a975697ee382e96fb99e8efd49c287bd367c8945a0d482`.
All 200 candidate and baseline promotion games used the same 200 unique seat/seed schedules
with zero errors. Candidate and parent won 59 and 44 games; their average VP differences
were -2.955 and -3.350. Despite the positive paired interval, exact comparison on the same
60-shard expanded plan found candidate-minus-parent regret deltas of about +0.00245 on
validation and +0.00281 on test. The final battery accounted for all 200 games and retained
the R/W/VP lower-bound gates, but `F=11/50` remained far below the absolute gate. Preserve
the data, retain checkpoint `b366433c5c7`, and do not promote `ed9b87946bb2`. The exact
expanded-holdout artifact SHA-256 is
`ffa622fdda6d4e810dc2e4aa77375e462c9f9c5c97027d6a05e7182740dd8519`.

The `32` treatment reused the same 60-shard aggregate and all training hyperparameters but
initialized from retained checkpoint `b366433c5c7`. The unchanged parent was measured as
epoch 0 and remained the fallback; epoch 10 was selected. Against the parent on exactly
61,435 validation and 61,804 test rows, candidate-minus-parent regret deltas were
-0.010813 and -0.010591, while legal-choice accuracy deltas were +3.11 and +2.96 points.
The n=200 round-2 paired point estimate also favored the child, including a +0.555 average
VP-difference gain, but its bootstrap lower bound was -4 points. The final battery retained
all weak gates and accounted for all 200 games with zero errors; `F=11/50` and -2.68 VP
still failed the absolute gate. This justified an independent confirmation, not promotion
from round 2 alone. Matched-holdout and paired artifact SHA-256 values are
`e054bdb3120107893a50fb21dd996dc9be54991333da0ae3a689d4cc4e87fbb2` and
`0fdf40e79842cfb5bab7c0e7727af1aec4d390f1d477ec940c9dce85f08d23fd`.

The predeclared `33` confirmation changed only the locked promotion seed round and game
count. On 800 unique shared seat/seed schedules, checkpoint `32031687acee` won 259 games
(32.375%) against `F` versus 223 (27.875%) for parent `b366433c5c7`. The paired score delta
was +4.5 points with a 95% bootstrap interval of `[+0.375, +8.75]` points, and average VP
difference improved from -3.10 to -2.45125 (+0.64875). Both reports contain exactly 800
accounted games, 400 in each seat, with zero truncations or errors. UCL workers shut down
before the confirmation could launch there, so it ran locally only after the first 20
round-2 schedules for both checkpoints exactly reproduced the remote outcome, VP, turn,
and tick evidence. The confirmation's own lower bound clears zero, satisfying the decision
fixed before launch: retain `32031687acee` as the supervised parent for iteration 2. This
relative promotion does not pass the absolute `F >= 52%` gate. The paired artifact SHA-256
is `f7476b1316e1efbd811430241cd59d661a63b423ceb34a475be26d4e1244b881`;
candidate and parent compact artifact file hashes are
`6d185dc872c76378fe875f1603fdaf14219a62b55f53cb4bcf69a308b061ce5d` and
`e341bd77b93020ec99cdf52cc786471debee775cfedeebe71cbf1d537bef596f`.

Iteration 2 (`34`) collected 100 alternating-seat parent-versus-`F` games: 10,124 rows in
10 shards, 36 diagnostic student wins, zero truncations, and an 84.68% exact teacher-action
match rate. Its immutable iteration-manifest SHA-256 is
`a6b4bce3c42ac88b922328056b826384804781a5a1d92241c8d64d473994851b`.
The `35` conservative replay used the exact 70-shard aggregate, parent initialization, and
epoch-0 fallback. Epoch 8 lowered validation regret from 0.071140 to 0.069442 and test regret
from 0.070603 to 0.068726. Round 4 was 55/200 (27.5%) versus 49/200 (24.5%), a +3-point
estimate with 95% interval `[-4, +10]`; its complete final battery was R/W/VP/F
100/100/100/30%. This justified one independent confirmation, not promotion.

The n=1600 round-5 confirmation (`36`) was sized and fixed before launch from round 4's
paired variance; its own interval was the only decision statistic. Candidate `7ec5729d2d47`
won 480/1600 games (30.0%) and retained parent `32031687acee` won 487/1600 (30.4375%) on
the same 1,600 unique shared schedules, 800 in each seat. The paired delta was -0.4375
points with 95% bootstrap interval `[-3.25, +2.4375]`; average VP difference was -2.8525
for the child and -2.604375 for the parent. All 3,200 reports were accounted with zero
errors or truncations. Reject the child and retain `32031687acee`. The paired artifact
SHA-256 is `e26f26771c00ac08d74e3bb60f6863b00191c42593327bcbe632404805184040`;
candidate and parent compact artifact file hashes are
`26726a7cd7b9c9cfdcfde5fbc45129e23a6bcf997f1c6934cff23a2ade019e6a` and
`aec98a7d317c84bd53968cc9ab742c0d80b25d107a051b3d9e6a0fcaacca1dcb`.

Run `37` exposed a real mixed-corpus bug: distillation shards omit the redundant
`ACTION_TYPE` column, while hard-state weighting assumed it existed. The preserved failed
run records `KeyError_ACTION_TYPE_in_mixed_distillation_corpus`; the loader now derives the
action family from the canonical action index when needed. The clean restart (`38`) showed
that the treatment itself is not useful: every post-update epoch worsened regret, epoch 0
was selected, and the exact matched report had zero deltas. The strict offline gate stopped
before consuming round 6. Its report SHA-256 is
`6fa57668f56571c84d9ad1ed71472d95d6e0ba9d4bdc0b4b2bf7d45bf53789c7`.

Finally, run `39` re-evaluated factored checkpoint `6dc7528ed909` against the retained MLP
on the exact 70-shard plan. The factored model's candidate-minus-parent mean-regret deltas
were +0.005570 validation and +0.004548 test; legal-choice accuracy fell by 2.041 and 2.037
points. This closes the old factored checkpoint as an immediate alternative. The reproducible
action-family breakdown combines the retained MLP's validation/test rows: `BUILD_ROAD`
accounts for 18,881 choices, only 46.63% legal-choice accuracy, and 4,079.58 total normalized
regret. The next contributors are `MOVE_ROBBER` at 1,658.34 and `MARITIME_TRADE` at 847.07.
This identifies a fresh-audit hypothesis, not a promotion claim. The action-family report
SHA-256 is `4a87fec0544f6377c2ffa0d7a567a012d690f4b6e8326c9e41d67ce7efaaad8b`.

Run `40` then froze a genuinely external road audit before designing the treatment: 100
alternating-seat parent-versus-`F` games, 9,522 decisions in 10 shards, zero truncations,
and iteration-manifest SHA-256
`d64e3be2b404df07e516c980560d7553d75040c8ac076f39c922c2c962d8d520`. All 13 audit files
were copied to UCL storage and verified byte-for-byte. The corpus was prohibited from
training, epoch selection, and hyperparameter tuning.

Run `41` loaded retained MLP `32031687acee`, froze all 747,852 base parameters, and trained
97,153 topology-aware residual parameters that can alter only road logits. The valid run
matched the declared 70-shard hash and 497,532/62,417/62,990 row split; a prior 30-shard
CLI-misgrouping attempt was quarantined before audit access. Selected epoch 3 improved
candidate-minus-parent mean regret by -0.003714 validation and -0.003471 test. On the two
untouched audit halves, aggregate regret improved by -0.001721/-0.003316; road total regret
fell by 8.17 and 14.25 while road accuracy rose by 0.145 and 0.293 points. All six declared
offline predicates passed. The locked 200-game round-6 comparison did not: the candidate won
54/200 (27.0%) against `F`, versus 51/200 (25.5%) for the parent, paired +1.5 points with
95% bootstrap interval `[-4, +7]`; VP-difference gain was +0.095. Reject checkpoint
`d7b106bd603cedf985e48d4e5b45ca89d48ad145ca17c51a4afd83bcf601b546`, retain
`32031687acee`, and do not extend the same observed schedule. The core, audit, and paired
report SHA-256 values are `5ca96dfcdf03`, `49a644c00129`, and `fb6d19b0abab`.

Run `42` added an exact matched-teacher diagnostic without authorizing training. Teacher RNG
was isolated from the six-game behavior trajectory, expensive labels were requested only when
`BUILD_ROAD` was legal, and the comparator verified 98 identical game/decision keys, state
hashes, legal sets, and behavior actions. Two M200 seed rounds agreed on 70.41% of exact labels,
disagreed with `F` on 77.55%, and stayed below the 500 ms p95 gate, so only an eight-game
deeper-quality pilot was authorized.

Run `43` stopped that branch. On 129 exactly matched states, the two M800 seed rounds agreed
on 72.09% of labels versus the 75% gate, and on only 50% of exact actions conditional on both
selecting `BUILD_ROAD` versus the 70% gate. Their worst p95 latency was 2052.96 ms versus the
1500 ms limit. The cheap M200 proxy agreed with M800 on 67.44% versus 75%. M800 did supply
different labels from `F` (80.62% disagreement), but novelty alone was insufficient. No search
road corpus, new audit, model training, or promotion schedule was authorized. Runs `42` onward
remain local because the remote approval service reached its usage limit; unlike runs `40` and
`41`, they are not claimed as UCL-preserved artifacts.

Run `44` then froze the declared contingency before model design: 100 alternating-seat
parent-versus-`F` games yielded 889 `MOVE_ROBBER` decisions in 10 verified shards with zero
truncations. Its iteration-manifest SHA-256 is
`0631355469934bb1ace6571c115adf459a2ddea53ce4389cde034c1846d60609`; the corpus was barred
from training, epoch selection, and hyperparameter tuning.

Run `45` loaded parent `32031687acee`, froze it, and trained 93,185 destination-aware residual
parameters that can alter only the 57 robber logits. Epoch 1 matched the declared 70-shard
hash and row split and improved candidate-minus-parent validation/test regret by
-0.0024269/-0.0019386. On the untouched audit halves, MOVE_ROBBER total regret fell
11.1573 to 9.7499 and 17.4440 to 16.8895; legal-choice accuracy rose 1.80 points on validation
and was unchanged on test. All six offline gates passed before round 7 was allocated. The
locked 200-game comparison rejected the checkpoint: 62/200 wins (31.0%) versus 65/200
(32.5%) for the parent, paired -1.5 points with 95% bootstrap interval `[-6, +2.5]` and
VP-difference delta -0.065. Retain `32031687acee`. Matched, audit, and paired report SHA-256
values are `2f2ad4fb644c`, `954d80f672d2`, and `705031922b5a`; compact candidate/parent file
hashes are `011e39e9a6da` and `cbcda0b0db0d`.

Run `46` audited the exact 70-shard, 622,939-row corpus before critic training. Every row had
a win target: 594,480 directly and 28,459 recovered from immutable DAgger game manifests.
VP-margin targets covered 594,480 rows (95.43%). The two base seat views were treated as one
logical corpus, while actual trajectories were namespaced and kept within 2,300 whole-game
split groups; no cross-logical-corpus collision was found. The minority win class was 38.53%.
All eight declared integrity/coverage gates passed. Public-score baselines over the combined
corpus were AUC 0.845427, sigmoid Brier 0.164429, non-tie accuracy 0.811487, margin MAE
4.472251, and margin RMSE 5.713699. The audit SHA-256 is
`5c6172e657e8765a44bd54c124aa1de5a11077144df96307b4c57fd291841471`.

Run `47` trained only a 235,842-parameter factored critic on the frozen
497,532/62,417/62,990 train/validation/test rows; the retained policy had zero trainable
parameters. Epoch 3 was selected. Validation AUC/Brier were 0.889341/0.133836 versus public
baselines 0.843571/0.166805, and margin MAE/RMSE were 2.779653/3.874191 versus
4.441142/5.667285. Test AUC/Brier were 0.885346/0.137315 versus 0.821522/0.180225, and
margin MAE/RMSE were 2.631524/3.678522 versus 4.518081/5.764710. All eight predeclared
gates passed. Checkpoint SHA-256 is
`bb3cf7012fe8d98622dd0e70a2275b1685c6af242c7ccd8a5d3fa342e77b6b1a`; offline-report
SHA-256 is `9f2af557be330d424f0b20db90012898e775c4b84540cc1053a954fc648b51c7`.

Run `48` fixed the only authorized policy-use design before gameplay: frozen run-32 policy,
frozen run-47 critic, policy top-k 3, generic chance-spectrum expectation, and a 0.05 minimum
expected-win improvement. Over 20 alternating-seat diagnostic games it observed 1,582 choice
decisions, scored 4,506 candidates, and reranked 151 times (9.54%). Mean/p95/max latency was
1.020/2.282/9.122 ms, with zero errors or truncations; its six then-declared mechanical gates
passed, but those gates omitted the hidden-information boundary later found to be violated.
The manifest and diagnostic SHA-256 values are `d4f06bc623a31a22bf677d585871299f61fbc61f8f1adec1989144168a8dbeec`
and `6e54bf8c87febfe4ad32a20e5d7972b0d5129b8797416de09c17bd1881a99b91`.

The locked round-8 n=200 comparison (`49`) was the promotion decision, not the diagnostic
win count. The reranker won 72/200 games (36.0%) against `F`; the same frozen policy without
reranking won 60/200 (30.0%). The paired delta was +6 points, but its 95% bootstrap interval
was `[-2, +14.5]`, failing the strict lower-bound-above-zero rule. VP difference improved
from -2.55 to -2.37 (+0.18). Even before the later invalidation, the exact reranker did not
pass promotion; retain checkpoint `32031687acee`. The paired report SHA-256 is
`2478e7c3fd17e6963410f98a9ede25a8d128ec1d940ca7811ba3ab64834a55b5`;
compact candidate/parent file hashes are `2f34f8a7ca60` and `cf12b3b9827f`. A 2026-08-31
audit later established that the generic successor spectrum used actual hidden opponent
resources for robber outcomes and exact hidden Monopoly transfers. The run-48/49 gameplay
evidence is invalidated. The implementation now uses public-only successors and falls back
to the frozen policy for action sets outside that boundary; no corrected gameplay run has
been made.

Run `50` then exercised the predeclared fresh-outcome contingency without reusing round 8.
It collected 200 alternating-seat run-32-vs-`F` games under seed `20260830`: 19,247 rows in
20 immutable shards, 61 wins, 139 losses, no draws or truncations. Only those fresh TRAIN
rows received the fixed multiplier `1 + max(0, -RETURN) + 0.5 * clip(max(0, -margin)/10,
0, 1)`; observed weights ranged from 1.0 to 2.5 with mean 1.78289. The manifest SHA-256 is
`61cd5127909f544f9ebb7d17233b9c181fec877d8b9d483cd65387dcac67f6c9`.

Epoch 5 checkpoint `f44e95993476` passed the exact 80-shard offline gate: candidate-minus-
parent mean-regret deltas were -0.006707 validation and -0.005894 test, while legal-choice
accuracy rose 0.55 and 0.72 points. That result authorized round 9 but did not predict it.
On 200 unique paired schedules, the candidate won 53 games (26.5%) against `F`, versus 73
(36.5%) for retained run 32. The paired delta was -10 points with 95% interval
`[-18.5, -2]`, and VP difference worsened by -0.74. The final battery retained R/W/VP point
gates at 100/100/96% but reached only 32% against `F`. Reject the checkpoint, allocate no
confirmation, do not tune the weights on this corpus or round, and retain `32031687acee`.
Matched-holdout and paired-report SHA-256 values are `d441ec90cef5` and `9dd061b3fc54`.

Run `54` tested the first human-visible same-turn PUCT treatment: 32 simulations, run-32
policy priors, the frozen run-47 outcome critic, no chance-spectrum calls, and no opponent-turn
expansions. Its 20-game diagnostic searched 988 choices, reached multiple plies on 713,
changed 3.74%, and ran at 16.05 ms p95 with zero errors, truncations, or information-boundary
violations. On fresh dev round 54 it scored 35/100 against `F` versus 33/100 for run 32, a
+2-point delta below the locked +3 requirement; VP difference improved +0.28. Reject the
critic leaf treatment without spending promotion round 14. Manifest, diagnostic, and paired
SHA-256 values are `1c398828cc4d`, `414533bc5b1c`, and `79e0147f4e34`.

Run `55` changed only the leaf to a public F position value that is exactly invariant to the
opponent's hidden resource composition. The policy, 32 simulations, sqrt(2) PUCT constant,
visibility set, and move rule stayed fixed. Its diagnostic searched 894 choices, reached
multiple plies on 615, changed 1.79%, and ran at 16.04 ms p95 with zero boundary violations.
Dev round 55 passed at 38% versus 32% with +0.95 VP-difference improvement. The locked
400-game promotion then passed strictly: 35.75% versus 32.25%, paired +3.5 points with 95%
interval `[+0.25, +6.75]`, and VP-difference delta +0.525. The final safety battery completed
all 200 games at R/W/VP/F = 96/98/100/38%; the weak floors passed, while the separate
absolute `F >= 52%` gate remains unmet. Retain manifest `36e4707e7621` as the playable bot
over frozen run 32. Diagnostic, promotion, and final report SHA-256 values are
`98496ea8295b`, `ed18d345c2eb`, and `aa5d712d0777`.

Run `56` widened only the retained run-55 search boundary to public chance nodes for dice and
development-card purchases. It kept the policy, public-F leaf, 32 simulations, sqrt(2) PUCT
constant, and final-move rule frozen. Its custom successor builder never called the generic
omniscient spectrum: dice outcomes used public controller state, while development-card
outcomes used the deck plus opponents' hidden unplayed-card counts as one unseen pool. Robber
movement, Monopoly transfers, and opponent turns remained unexpanded. The 20-game diagnostic
passed all 12 gates: 1,639 searches, 1,113 multi-ply decisions, 4,050 chance actions and 20,669
chance outcomes expanded, 1.04% policy changes, 20.67 ms p95, and zero errors, truncations,
probability violations, forbidden expansions, or opponent-turn expansions. Dev round 56 passed
at 35% versus 30%, paired +5 points and VP-difference delta +0.27. Locked promotion round 16
accounted for all 400 games with zero errors: 34.50% versus 32.25%, paired +2.25 points with
95% interval `[-0.25, +4.75]`, and VP-difference delta +0.3525. Because the interval did not
strictly clear zero, reject manifest `7d22c13caa7f`, skip the final battery, perform no tuning
on the consumed round, and retain run 55. Diagnostic and promotion SHA-256 values are
`66c804e0c6f7` and `1cbdb86b3081`.

## Record a corrected result

```bash
python examples/colonist_1v1_evaluate.py \
  --agent L:runs/<run>/colonist_maskable_ppo.zip \
  --protocol milestone --gates \
  --eval-kind final --gate-mode lower_bound \
  --report runs/<run>/final_benchmark.json

python examples/colonist_1v1_publish_result.py \
  runs/<run>/final_benchmark.json \
  --output docs/results/<experiment>.json
```

Add a row only after the publisher accepts the report. A rejected model is still useful
evidence; a report with missing games, errors, development seeds, or no checkpoint hash is not.
