# Results log

This file separates historical observations from promotion-grade evidence.

> **Evidence reset — 2026-07-12.** Every evaluation below predates the repaired
> game-accounting path. The old accumulator could omit turn-limit games from the
> denominator and final-VP averages. Reports also predate reliable fixed-seat
> scheduling, so their `seat0/seat1` fields are invalid. All old numbers are
> **provisional historical estimates**, not current scorecards, promotion evidence,
> or proof that any checkpoint is the best or most seat-balanced model.

No corrected rebaseline or accepted compact result has been recorded yet. The next
evidence action is to re-evaluate retained checkpoints with a final seed suite, both
seats, lower-bound gates, complete per-game rows, and checkpoint hashes.

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

- retained policies probably have a material gap against `F`, but the corrected size of
  that gap is unknown;
- legal-choice and candidate-value objectives may preserve close-action preferences better
  than full-space imitation accuracy, but listwise BC has not yet been run and measured;
- search-distillation may help if repaired MCTS is both strong and affordable, but the
  required `05-mcts-strength-sweep` has not yet been run;
- old per-seat differences cannot support any seat-balance conclusion;
- full AlphaZero-style training is a gated fallback, not the established next solution.

## Corrected evidence ledger

| Date | Experiment | Checkpoint hash | Seed suite | Result | Artifact |
|---|---|---|---|---|---|
| - | Corrected historical rebaseline | - | `final` | Not run | - |
| - | `05-mcts-strength-sweep` | N/A | Held-out search seeds | Not run | - |
| - | Legal-CE BC baseline | - | `promotion` / `final` | Not run | - |
| - | Listwise BC treatment | - | `promotion` / `final` | Not run | - |
| - | DAgger/search distillation | - | `promotion` / `final` | Not run | - |
| - | GPU backlog experiments | - | `promotion` / `final` | None completed | - |

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
