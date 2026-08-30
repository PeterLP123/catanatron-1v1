# Plan: evidence-first path to a stronger 1v1 bot

> **Current as of 2026-08-30.** DAgger iteration 0 is the best measured policy.
> Iteration 1 is discarded. Executable GPU queue definitions live in
> `catanatron.gym.experiment_backlog`, the generated view is
> [GPU_EXPERIMENT_BACKLOG.md](GPU_EXPERIMENT_BACKLOG.md), and accepted evidence
> belongs in [RESULTS_LOG.md](RESULTS_LOG.md) and [`docs/results/`](results/README.md).

## Evidence reset

Every model result recorded before the 2026-07-12 evaluation-accounting repair is
**provisional**. Turn-limit games could disappear from the denominator and final-VP
statistics, and older seat labels were not reliable. Those reports can motivate a
hypothesis, but they cannot promote a checkpoint or establish a current best model.

The immediate goal is not another long PPO run. Corrected evidence showed that the
hybrid-BC parent has a real but incomplete signal against `F`, while PPO—including a broad
forward-KL anchor sweep—rapidly removes it. One `F`-labelled student-visited DAgger
iteration is kept. A second 100-game iteration did not improve F rate or VP margin, so
the DAgger line stops at iteration 0.

## What is implemented

- every requested evaluation game is now represented as a win, loss, draw/truncation,
  or error, with final VP and per-game schedule evidence;
- development, promotion, and final evaluation use disjoint deterministic seed suites;
- promotion/final gates can use the Wilson confidence lower bound, and paired reports
  can be compared on shared seat/seed schedules with a bootstrap interval;
- model artifacts carry feature, action, rules, and combined schema hashes;
- datasets and runs record shard/data hashes, Git state, Python, package, and hardware
  provenance;
- BC streams Parquet shards in batches, splits by whole game, supports CPU/CUDA/MPS,
  and can train with legacy full-space CE, legal-masked CE, or candidate-value listwise
  loss;
- the MCTS benchmark records latency and two-seat strength for held-out seed suites;
- the DAgger/search-distillation CLI can collect immutable student-visited datasets
  with deterministic F or fixed-simulation MCTS labels;
- the backlog has evidence-derived `accepted`, `rejected`, and `inconclusive` states;
- compact result publishing, reversible checkpoint archival, and CPU CI are available.

## What is not yet evidence

- the retained legacy checkpoints have not all been re-evaluated under repaired accounting;
- the completed teacher-population screen used only two games per matchup, so it is a
  directional diagnostic rather than teacher-promotion evidence;
- DAgger iteration 1 was measured and discarded: F 34% / -2.18 VP versus kept iteration 0 at 36% / -1.80 VP;
- no candidate has retained both an `F >= 10%` signal and all weak gates after PPO;
- no 5M promotion or AlphaZero-style training has been justified or run.

## Current decision

The reconstructed hybrid-BC parent is the control: promotion-suite `R=98%`, `W=100%`,
`VP=96%`, and `F=20%` (VP difference `-3.18`) over 50 games per opponent. DAgger
iteration 0 is the current best policy: `R/W/VP=100/100/100%`, `F=36%` (18/50), VP
difference `-1.80`, held-out regret `0.080`. Iteration 1 is discarded: `F=34%`
(17/50), VP difference `-2.18`, regret `0.086`. Weak gates held, but F rate and VP
margin both moved the wrong way, so do not grow the DAgger corpus or retune the
augmentation weight on this promotion suite.

The F 52% gate still fails; published promotion reports remain rejected on that gate.
The older UCL hybrid-BC final-suite card (`F=24%`, `-2.50` VP) and the PPO/anchor
failures still stand. PPO retention stays rejected until a new one-variable treatment
is specified. The DAgger line has plateaued, so a later retention-gated PPO warm-start
from iteration 0 is now an allowed candidate, not an automatic next run.

Reconstruct a missing parent with `scripts/run_hybrid_bc_parent.sh`. Replay the first
pilot with `scripts/run_strong_bot_path.sh`. `scripts/run_dagger_f_next.sh` stays for
a future student that actually beats iteration 0.

## Execution order

### 0. Repair and freeze the measurement surface

The code work is implemented. Before using a result, require all requested games to be
accounted for, zero evaluator errors, one per-game row per request, a checkpoint hash,
explicit seat scheduling, and a named seed suite. Publishable evidence must be a locked
promotion or final report, not a development report.

### 1. Run the corrected rebaseline

Re-evaluate the retained historical checkpoints with the same final protocol, final seed
suite, both seats, and lower-bound gates. Record truncations and errors explicitly. This
step answers only what the existing models actually do under trustworthy measurement; it
does not retroactively validate the old seat splits or exact win rates.

The rebaseline becomes the control for every later branch. If an artifact cannot be tied
to its checkpoint hash and schema, record it as legacy context rather than promotion
evidence.

### 2. Measured MCTS; keep `F` as the practical teacher

The completed `05-mcts-strength-sweep` found only 5–10% wins against `F` at 100 ms, with
p95 latency around 283 ms. The later population diagnostic completed all 28 two-game cells:
`AB:2` was fast but lost both games to `F`; `M:800` and `M:2000` each split their two games
against `F` but cost about 1.22 s and 3.21 s p95 respectively. This sample is too small to
promote a search teacher and does not displace `F` for the bounded DAgger pilot.

Accept a search teacher only if the sweep is complete, reproducible, and stronger than
the reactive baseline at an affordable latency. A profile-only run is diagnostic and does
not satisfy this gate.

### 3. Hybrid BC established the supervised control

Create one locked train/validation/test split by whole game and compare:

1. legal-masked cross entropy as the baseline;
2. candidate-value listwise loss on the same scored decisions;
3. optionally the `public_derived` feature profile against the `raw` control.

Use deterministic seeds and the same architecture. Select the best epoch by held-out
regret when candidate values exist, otherwise by validation loss. Do not accept a model on
raw action accuracy alone. The listwise checkpoint must lower held-out regret and then
improve locked F/search outcome or VP-margin evidence over the legal-CE checkpoint.

The selected hybrid legal-CE/listwise checkpoint now supplies the control described above.
The next supervised comparison holds its base corpus, objective, architecture, and split
fixed while adding only the independently split DAgger corpus.

### 4. Distil `F` on states the student actually visits

Use `F`, the strongest measured practical teacher, for small DAgger/expert-iteration cycles:

1. let the current student generate its own visited states;
2. label each legal-action set with F or fixed-simulation MCTS;
3. verify immutable shard and manifest hashes;
4. aggregate the new iteration with prior data;
5. retrain with legal/listwise BC and evaluate on the locked promotion suite.

Start with tens of games, not a large generation job. Continue only while held-out regret
and locked gameplay evidence improve. The implemented CLI is a data-collection scaffold;
it does not claim that expert iteration has already succeeded.

### 5. Pause PPO until supervised improvement is accepted

PPO remains a refinement stage, not the source of a new hypothesis. The completed anchor
diagnostics failed to retain the hybrid-BC signal even at coefficient `10`, so do not launch
more PPO until DAgger or another supervised treatment improves the locked parent. Then hold dataset, model schema,
PPO hyperparameters, seed suite, and evaluation protocol fixed. Change one treatment at a
time, beginning with the actual-VP versus visible-VP reward pair only after the corrected
baseline and teacher/BC decisions are recorded.

Use development evaluation for local checkpoint selection only. Promotion and final
decisions require their disjoint seed suites and confidence-lower-bound gates. Run one
replication for the winning treatment; treat a paired per-game outcome-score delta below 0.03 as
inconclusive.

### 6. Scale only a credible signal

The 5M `30-strong-promoted` run remains locked behind at least 10% observed win rate versus
`F` plus retained `R >= 90%`, `W >= 70%`, and `VP >= 60%` gates in complete comparable
reports. This is an early-signal gate, not the final target. Preserve checkpoint, schema,
manifest, environment lock, and evaluation evidence before reversible archival.

### 7. Consider full AlphaZero only behind an evidence gate

Do not start a policy/value self-play rewrite merely because BC or PPO disappoints. Revisit
full AlphaZero-style training only if all of these are true:

- repaired MCTS supplies useful targets at a tolerable budget;
- legal/listwise BC and several DAgger iterations plateau under locked evaluation;
- controlled PPO cannot turn that teacher signal into further progress;
- the expected compute and implementation cost are explicitly budgeted.

If those conditions are not met, the correct result is a well-measured reactive policy and
a documented negative research outcome, not an unbounded training run.

## Decision gates

| Decision | Evidence required |
|---|---|
| Report is usable | All games accounted; zero evaluator errors; per-game rows, checkpoint hash, both seats, named seed suite |
| Historical checkpoint becomes the baseline | Corrected final-suite re-evaluation published; no legacy seat claims carried forward |
| Search can teach | Complete `05` sweep with required budgets/opponents/seeds and p95 latency |
| Listwise BC beats baseline BC | Lower held-out regret plus better locked F/search result or VP margin |
| DAgger iteration is retained | Verified immutable data and improvement over its parent on held-out regret and promotion evidence |
| PPO treatment wins | Paired gain at least 0.03, weak gates retained, direction repeated on one new seed |
| Candidate deserves 5M | F at least 10% and all weak gates retained in complete comparable reports |
| AlphaZero work begins | Useful search teacher, distillation plateau, PPO plateau, and explicit compute budget |

Stop any run on NaNs, repeated CUDA failures, full disk, evaluator errors, incomplete game
accounting, or no progress events for 15 minutes. Never promote from a development seed suite.

## Working documents

| Document | Responsibility |
|---|---|
| [RESULTS_LOG.md](RESULTS_LOG.md) | Historical context and corrected evidence ledger |
| [GPU_EXPERIMENT_BACKLOG.md](GPU_EXPERIMENT_BACKLOG.md) | Generated queue, commands, resources, gates, and stop rules |
| [TRAINING.md](TRAINING.md) | Data, BC, distillation, PPO, evaluation, and artifact reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Runtime boundaries and evidence flow |
