# Plan: GPU-gated path to a stronger 1v1 bot

> **Current as of 2026-07-03.** This document records the project direction,
> evidence, and promotion gates. The executable run order and commands live in
> [GPU_EXPERIMENT_BACKLOG.md](GPU_EXPERIMENT_BACKLOG.md); measured outcomes live in
> [RESULTS_LOG.md](RESULTS_LOG.md). If the documents disagree about an experiment,
> the backlog definition in `catanatron.gym.experiment_backlog` is authoritative.

## Where the project stands

The simulator and training platform are ready for the first GPU run. The current best
reactive policy, `runs/v2_ppo_fheavy/colonist_maskable_ppo.zip`, is strong against the
weak tiers and reasonably balanced between seats, but it is not competitive with the
hand-crafted lookahead tier.

| Opponent | Current best win rate | Required gate | Status |
|---|---:|---:|---|
| `R` | 97.4% | 90% | Pass |
| `W` | 85.3% | 70% | Pass |
| `VP` | 85.6% | 60% | Pass |
| `F` | 1.0% | 52% | Fail |
| `AB:2` | 0.5% | 52% | Fail |

The near-term deliverable is therefore only partially complete: the model convincingly
beats the random and reactive baselines, but the original goal of competitive play against
`F` and search opponents remains open.

Completed engineering work:

- two-seat evaluation with fixed seat order and deterministic seed schedules;
- behavioral cloning, MaskablePPO, mixed leagues, curricula, checkpointing, and strength reports;
- hard-state sampling, candidate-action scoring, held-out decision regret, and F-leaf evaluation;
- MCTS correctness fixes, transposition caching, latency budgets, and search profiling;
- reproducible UCL CS and Myriad GPU setup, launch, monitoring, and evaluation scripts;
- a dependency-gated experiment queue with explicit promotion and stop rules.

No GPU backlog experiment has completed yet. `00-gpu-smoke` is the next action.

## What the failed work established

Four materially different approaches all remained at roughly 0-1% against `F`:

| Approach | Result against `F` | What it ruled out |
|---|---:|---|
| Self-play PPO, 500k steps | 0.5% | More unanchored model-free training is not enough |
| BC on 5.4M samples | 0.5% | High imitation accuracy does not preserve decision quality |
| PPO trained directly against `F` | about 1% | Opponent exposure alone does not solve the gap |
| One-ply search with a learned value net | 0.5% | Search structure alone does not replace a useful value gradient |

The strongest diagnosis is a **decision-margin value problem**. Terminal-outcome prediction
is easy on late, obvious positions and can report high global accuracy while remaining too
flat or noisy to rank close legal actions in the early and middle game. `F` succeeds because
its hand-crafted value function supplies a useful preference at those decisions.

This evidence changes the execution rule: do not scale another PPO run merely because it
improves weak-opponent win rates. A candidate must first produce a real `F` signal or a
materially better `F` victory-point margin.

## Current execution plan

The stages below match the IDs and dependencies in the GPU backlog. Commands, resource
estimates, and storage guidance are intentionally not duplicated here.

### Stage 0: validate the GPU workflow

Run `00-gpu-smoke` on the intended UCL host. It must complete 20k steps, produce checkpoints
and a mid-run evaluation, feed the dashboard, and finish without a CUDA error. Until this
passes, the training platform is code-complete but not hardware-validated.

### Stage 1: establish the reward baseline

Run the matched seed-101 pair:

- `10-balanced-actual-s101`: current actual-VP shaping control;
- `11-balanced-visible-s101`: public-score-only shaping treatment.

Compare two-seat scorecards from the same commit and protocol. Visible-VP shaping wins only
if it improves weighted score by at least 0.03 without losing the `R`, `W`, or `VP` gates.
Treat a smaller difference as inconclusive.

Run only the seed-202 replication for the treatment that wins seed 101:

- `12-balanced-actual-s202`, or
- `13-balanced-visible-s202`.

The direction of the improvement must repeat. Do not run both replications automatically.

### Stage 2: test choice-focused behavioral cloning

Generate scored hard-state data, train a BC checkpoint with `--hard-states`, and inspect
held-out legal-choice accuracy and mean regret. Raw action accuracy is not the gate.

Run `20-hard-bc-actual-s101` only when the checkpoint lowers held-out decision regret. Run
`21-hard-bc-visible-s101` only if visible-VP reward won Stage 1. Keep the branch only if it
improves `F` win rate or `F` VP margin over the matched reward baseline while retaining the
weak gates.

### Stage 3: promote only a credible signal

Run `30-strong-promoted` for 5M steps only after a 500k candidate reaches at least 10% against
`F` and still passes `R >= 90%`, `W >= 70%`, and `VP >= 60%`. The 10% threshold is a
promotion signal, not the final strength target.

Evaluate a promoted candidate with the milestone and full protocols, including held-out search
opponents. Require the post-2026-07-01 per-seat results to remain credible; historical seat
splits from before the seat-order correction are not comparable.

### Stage 4: optional anchored self-play

Run `40-selfplay-polish` only after Stage 3 produces a strong anchored parent. Keep it only if
`F` or search strength improves and each weak-tier result stays within two percentage points of
the parent. Self-play is a finishing step, not a route around the promotion gate.

## Decision gates

| Decision | Evidence required |
|---|---|
| GPU workflow is usable | Smoke run completes with CUDA, checkpoints, evaluation, and dashboard output |
| Reward treatment wins | At least +0.03 weighted score, no weak-gate regression, then same direction on seed 202 |
| Hard-state BC is worth PPO time | Lower held-out regret before training; better `F` rate or VP margin after training |
| Candidate deserves 5M steps | At least 10% against `F` and all weak gates retained |
| Candidate meets the original strength bar | At least 52% against `F` and the required search opponents under two-seat evaluation |
| Self-play polish is accepted | Search strength rises; weak results stay within two points of the parent |

Stop a run on NaNs, repeated CUDA errors, a full disk, or no progress events for 15 minutes.
Do not infer a winner from one seed when the weighted-score difference is below 0.03. Preserve
the winning checkpoint, manifest, evaluation report, and registry before pruning artifacts.

## If the gated backlog fails

If the 500k candidates remain in the existing 0-4% `F` band, stop the current PPO track. The
next credible research direction is AlphaZero-style policy/value learning with iterated
self-play and MCTS-generated policy and value targets. That system is not implemented in the
current backlog, requires a separate design and compute budget, and has uncertain payoff.

Do not quietly turn that stretch track into another long PPO run. The failed experiments already
show that scale without decision-margin signal is not a justified use of GPU time.

## Scope and non-goals

- The project remains a local simulator and trainer. It does not connect to or automate Colonist.
- "Human-like" is measured through reactive play, diverse opponents, seat balance, public-state
  ablations, and held-out evaluation; no human-game dataset is currently in scope.
- Search opponents are evaluation anchors. The preferred deployed policy remains reactive unless
  the project explicitly adopts the AlphaZero research track.
- Generated data, checkpoints, and reports remain under ignored `data/` and `runs/` directories;
  comparable outcomes must be recorded in [RESULTS_LOG.md](RESULTS_LOG.md).

## Working documents

| Document | Responsibility |
|---|---|
| [RESULTS_LOG.md](RESULTS_LOG.md) | Recorded evidence and comparable scorecards |
| [GPU_EXPERIMENT_BACKLOG.md](GPU_EXPERIMENT_BACKLOG.md) | Run order, commands, resources, dependencies, and stop rules |
| [TRAINING.md](TRAINING.md) | General training and evaluation reference |
| [UCL_CS_GPUS.md](UCL_CS_GPUS.md) | Interactive UCL CS GPU-host workflow |
| [UCL_MYRIAD.md](UCL_MYRIAD.md) | Scheduled Myriad workflow |
