# Plan: evidence-first path to a stronger 1v1 bot

> **Current as of 2026-09-04.** Run 55 is the retained playable wrapper over the frozen
> run-32 supervised policy. Runs 56–58 are not retained, and the absolute `F >= 52%` target remains
> unmet. Evidence status for every completed or proposed experiment is stated explicitly
> below. Executable GPU queue
> definitions live in `catanatron.gym.experiment_backlog`, the generated view is
> [GPU_EXPERIMENT_BACKLOG.md](GPU_EXPERIMENT_BACKLOG.md), and accepted evidence
> belongs in [RESULTS_LOG.md](RESULTS_LOG.md) and [`docs/results/`](results/README.md).

## Evidence reset

Every model result recorded before the 2026-07-12 evaluation-accounting repair is
**provisional**. Turn-limit games could disappear from the denominator and final-VP
statistics, and older seat labels were not reliable. Those reports can motivate a
hypothesis, but they cannot promote a checkpoint or establish a current best model.

The immediate goal is not another long PPO run or another uniform DAgger iteration.
Corrected evidence shows that the hybrid-BC parent has a real but incomplete signal against
`F`, while PPO—including a broad forward-KL anchor sweep—rapidly removes it. Two bounded
student-visited DAgger cycles produced one independently confirmed improvement: checkpoint
`32031687acee` beat its parent by +4.5 points on a predeclared n=800 paired round. Later
uniform replay, generic hard-state weighting, and two narrowly spatial action-family heads
all failed to improve locked gameplay. The road and robber heads did improve exact matched
and fresh-audit imitation metrics, so the failure is not simply an inability to fit those
labels. A fixed-search road teacher also failed its predeclared stability, agreement, and
latency gates. Treat action imitation of `F` as a measured plateau.

The outcome branch has now answered that question. Run `46` established complete win-target
coverage, 95.43% VP-margin coverage, and leakage-safe whole-game groups across the frozen
70-shard corpus. Run `47` trained a separate critic that beat public-score baselines on all
eight validation/test gates without changing the policy. One fixed top-3, 0.05-threshold
reranker appeared to pass its operational gate and produced an inconclusive +6-point paired
estimate against `F` in round 8 (95% CI `[-2, +14.5]`). A 2026-08-31 code audit then found
that its generic chance spectrum inspected hidden opponent resources for robber/Monopoly
successors. Runs `48` and `49` are therefore invalidated, not promotion evidence. The wrapper
now uses only public successors and falls back to the frozen policy outside that boundary;
the corrected wrapper is untested in gameplay. Run `32` remains retained. Outcome-aligned
learning still needs genuinely new
student-visited outcome data or a predeclared advantage-learning design before another
gameplay schedule—not post-hoc threshold changes.

A clean repository-local reconstruction separately confirmed the early DAgger boundary:
iteration 0 improved promotion-suite `F` from 20% to 36%, iteration 1 fell to 34%, and
retention-gated PPO collapsed to 2% at its 10k stop. Those reproducibility artifacts do not
override the later paired sequence or justify another PPO pass.

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
- BC can alternatively use a factored edge/node/tile/port/global encoder, learned action
  embeddings, and supervised win/VP-margin heads while old MLP checkpoints remain loadable;
- BC can wrap a retained MLP with zero-initialized topology-aware residuals that change only
  `BUILD_ROAD` or `MOVE_ROBBER` logits, freeze the byte-compatible base, and load the result
  normally;
- BC can initialize from a schema-compatible parent, include the unchanged parent as
  epoch 0, and compare child and parent on one exact expanded holdout before gameplay;
- BC corpus options accumulate when repeated, and optional expected SHA/shard/split-row
  contracts abort drift before the first optimizer step;
- a single command can run candidate and baseline on identical schedules and write a paired
  bootstrap promotion artifact;
- the MCTS benchmark records latency and two-seat strength for held-out seed suites;
- the DAgger/search-distillation CLI can collect immutable student-visited datasets
  with deterministic F or fixed-simulation MCTS labels, isolate teacher RNG, filter before
  expensive teacher calls, record latency and terminal outcome targets, and compare labels on
  exact matched states;
- an outcome audit recovers legacy manifest labels, preserves logical-corpus split groups,
  measures target coverage/class balance, and computes public-score baselines;
- a value-only factored critic trains on the same frozen whole-game splits, and a composite
  `C:` player can rerank a frozen policy with public-only successors and a policy fallback,
  while verifying both weights and all sidecar hashes;
- the backlog has evidence-derived `accepted`, `rejected`, and `inconclusive` states;
- compact result publishing, reversible checkpoint archival, and CPU CI are available.

## What is not yet evidence

- the retained legacy checkpoints have not all been re-evaluated under repaired accounting;
- the completed teacher-population screen used only two games per matchup, so it is a
  directional diagnostic rather than teacher-promotion evidence;
- DAgger iteration 1 was measured and discarded: F 34% / -2.18 VP versus kept iteration 0 at 36% / -1.80 VP;
- retention-gated PPO from DAgger-0 was measured and rejected: 10k stop, promotion F 2%;
- no post-retention treatment has improved both matched held-out regret and independent
  paired gameplay over checkpoint `32031687acee`;
- the outcome critic is useful offline, but the only frozen-policy reranker gameplay run is
  invalidated by hidden-information leakage and is not evidence of a stronger bot;
- no candidate has retained both an `F >= 10%` signal and all weak gates after PPO;
- no 5M promotion or AlphaZero-style training has been justified or run.

## Current decision

The reconstructed hybrid-BC control scored promotion-suite R/W/VP/F = 98/100/96/20%.
Its first DAgger iteration reached 100/100/100/36%, while iteration 1 fell to 34% and
retention-gated PPO from iteration 0 collapsed to 2%. Preserve those artifacts as a local
reproduction, but use the later paired sequence below for the retained lineage decision.

Retain checkpoint `32031687acee2636e60a2b2c9bf667580f7a884e5e223ddc230d131726381943`.
Its first n=200 comparison was inconclusive, but the predeclared independent round 3
reproduced the +4.5-point gain: 259/800 wins (32.375%) against `F` versus 223/800
(27.875%) for parent `b366433c5c7`, with a 95% paired interval of `[+0.375, +8.75]`
points and a +0.64875 VP-difference gain. This is a defensible relative promotion, not an
absolutely strong bot: 32.375% still fails the `F >= 52%` gate.

Iteration 2 added 100 alternating-seat games and 10,124 immutable teacher-labelled rows.
The parent-initialized child improved validation/test regret from 0.07114/0.07060 to
0.06944/0.06873. Its round-4 n=200 point estimate was +3 points, but the fixed n=1600
round-5 confirmation reversed direction: 30.0% versus 30.4375%, paired -0.4375 points
with 95% interval `[-3.25, +2.4375]` and VP-difference delta -0.248125. Reject checkpoint
`7ec5729d2d47` and retain `32031687acee`.

Generic hard-state weighting is also rejected. Every trained epoch regressed; the epoch-0
fallback exactly reproduced the parent and failed the strict offline-improvement gate before
using a new promotion schedule. The existing factored checkpoint is not a hidden solution:
on the 70-shard expanded holdout it worsened validation/test regret by +0.00557/+0.00455
and legal-choice accuracy by about two points relative to the retained MLP.

The reproducible error decomposition now directs the next experiment. Across the 70-shard
validation/test holdout, `BUILD_ROAD` contributes 4,079.58 total normalized regret over
18,881 choices at only 46.63% accuracy; `MOVE_ROBBER` is next at 1,658.34. Simple
family-wide reweighting hurts. Run `40` froze 9,522 decisions from 100 parent-visited games;
run `41` used an edge-structured residual and passed every declared offline predicate. On the
two untouched audit halves, road regret fell 5.5% and 8.6%, while aggregate regret also fell.
Fresh promotion round 6 was only 27.0% versus 25.5% against `F`: paired +1.5 points with 95%
interval `[-4, +7]`. Reject `d7b106bd603c` and retain `32031687acee`.

After run `41`, the fixed next decision was to avoid tuning its consumed audit and benchmark
whether a deterministic fixed-simulation search labeler supplied stable, affordable,
meaningfully different road rankings on training-only states. Failure would move exactly once
to the next measured error class (`MOVE_ROBBER`) instead of scaling road imitation.

That contingency is now complete. Run `42` validated the matched-label mechanism, but the
larger run `43` rejected the search-road branch: M800 round agreement was 72.09% versus a
75% gate, exact agreement when both chose roads was 50% versus 70%, p95 label latency was
2.05 s versus 1.5 s, and M200 agreed with M800 only 67.44% versus 75%. No search-labelled
training corpus or promotion schedule was authorized.

Runs `44` and `45` then exercised the predeclared `MOVE_ROBBER` contingency. The fresh audit
contained 889 choices from 100 alternating-seat games and was frozen before architecture
work. The 93,185-parameter frozen-base robber residual passed all six offline gates: matched
validation/test regret improved by -0.002427/-0.001939, and untouched audit regret improved
without an accuracy loss. Locked round 7 rejected it anyway: 31.0% versus 32.5% for the
retained parent, paired -1.5 points with 95% interval `[-6, +2.5]`. Reject
`dc58347a6642` and retain `32031687acee`.

Stop action-family residuals here; `MARITIME_TRADE` is not the next automatic treatment.
Run `46` audited all 622,939 rows: win coverage was 100%, VP-margin coverage was 95.43%,
and 4,300 trajectories remained in 2,300 leakage-safe split groups. Run `47` selected epoch 3
of a 235,842-parameter value-only critic. On test it improved AUC from 0.82152 to 0.88535,
Brier score from 0.18023 to 0.13731, margin MAE from 4.51808 to 2.63152, and margin RMSE
from 5.76471 to 3.67852; all eight validation/test gates passed.

Run `48` then fixed top-k `3`, minimum expected-win improvement `0.05`, a generic chance
expectation, and the frozen run-32 policy and critic. Its operational diagnostic and locked
round-8 result (`49`) were later invalidated when a 2026-08-31 audit found hidden opponent
resources entering robber/Monopoly successors. The historical scores remain recorded, but
they are not evidence for the corrected public-only wrapper. Retain `32031687acee`.

Do not extend round 8 or tune top-k/threshold on it. The next defensible experiment must add
new information: collect a predeclared, immutable corpus of fresh run-32-visited trajectories
using the now-native win and VP-margin targets, audit its distribution and split contract,
then test one outcome/advantage treatment on a new schedule. Until that passes, the champion
and absolute-strength conclusion are unchanged.

That treatment completed as run `50-loss-conditioned-dagger-s20260830`: 200 fresh
alternating-seat run-32-vs-`F` games, collection seed `20260830`, immutable ten-game shards,
and one loss-conditioned imitation rule. Fresh training rows keep unit weight after wins;
losses add `1.0`, while terminal VP deficits add at most `0.5`, saturating at ten points.
The exact base and iteration 0/1 corpora, augmentation multiplier `4`, MLP, hybrid objective,
warm start, seed `101`, learning rate `0.001`, and ten epochs stay fixed. Missing or non-finite
native targets abort the run. Expanded validation/test regret must improve before untouched
paired round `9` is consumed; promotion then requires a paired 95% lower bound above zero
without losing the final weak-opponent gates. No coefficient tuning is admissible on this
corpus or round.

Run `50` passed that offline gate but failed gameplay decisively. Its selected epoch improved
expanded validation/test regret by -0.006707/-0.005894 and legal-choice accuracy by
+0.55/+0.72 points. Locked round 9 then produced 26.5% versus 36.5% for run 32, paired
-10 points with 95% interval `[-18.5, -2]` and VP-difference delta -0.74. Reject the
checkpoint and treat terminal-loss weighting as another case where better imitation metrics
did not transfer to strength. No confirmation or coefficient tuning is authorized; run 32
remains the champion.

Runs `51`–`53` then tested three non-weighting mechanisms. The fixed setup-only `F` wrapper
passed its operational and development gates and reached 36.0% versus 29.75% in locked round
11, but its +6.25-point interval was `[0, +12.5]`; the strict lower bound did not clear zero.
The direct shared action-conditioned architecture failed before gameplay, worsening matched
validation/test regret by +0.01738/+0.01660 and legal accuracy by more than five points. A
lineage-verified 50/50 weight soup of run 32 and its run-35 child improved expanded-holdout
regret and accuracy, but locked round 13 was only 30.0% versus 26.0%, paired +4 points with
interval `[-1, +9]`. Reject all three exact treatments without routing, embedding, or
interpolation tuning. Run 32 remains champion; the positive-but-inconclusive wrapper and soup
results are hypotheses for a future independently motivated design, not promotions.

Runs `54`–`56` then tested hidden-information-safe same-turn search. The outcome-critic leaf
missed its development gate. The public-F leaf cleared promotion round 15 over run 32:
35.75% versus 32.25%, paired +3.5 points with interval `[+0.25, +6.75]`, and retained the
weak final battery at R/W/VP/F = 96/98/100/38%. Retain run 55 as the playable wrapper over
run 32. Run 56 added public dice and development-card chance nodes, but its round-16 gain
was inconclusive: 34.50% versus 32.25%, paired +2.25 points with interval
`[-0.25, +4.75]`. Reject run 56 without tuning. The absolute `F >= 52%` gate remains unmet.

Run `57` added only a bounded own-hand construction-readiness bonus to run 55's
public-F leaf. The pilot passed: 38% versus 35% against F, plus 54/100 wins in a
direct match against run 55. Fresh 400-game-per-bot confirmation was inconclusive:
30.0% versus 29.75%, paired +0.25 points with 95% interval [-1.75, +2.25], and VP
margin -0.0325. All 1,120 evaluation games completed without errors or truncations.
Retain run 55, skip the conditional final battery, and do not tune v1 on these
consumed schedules. See the [fixed specification and results](experiments/57-own-hand-puct.md).

Run `58` changed only run 55's simulation budget from 32 to 128. Both operational
diagnostics passed, but p95 search latency rose from 35.88 to 82.03 ms on matching
seeds. The fresh F pilot scored 41/100 versus 39/100, paired +2 points with 95%
interval [-3, +8], below the predeclared +3-point development gate. The direct
score was 53–47 with +0.31 VP margin. All 340 games completed without errors or
truncations. Retain run 55; confirmation, the final battery and root-visit
distillation were not run. Promotion/final round 18 remain unconsumed. See the
[fixed budget comparison and decision](experiments/58-search-budget.md).

Reconstruct a missing early parent with `scripts/run_hybrid_bc_parent.sh`; replay the first
pilot with `scripts/run_strong_bot_path.sh`. `scripts/run_dagger_f_next.sh` remains only for
a future student that clears its frozen parent on new evidence.

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

The first retained update and its independent confirmation are complete. Iteration 2 is
also complete: preserve its immutable data, but reject its child after the n=1600 paired
confirmation. A hard-state-weighted replay treatment failed offline and consumed no new
promotion schedule. Do not collect more uniformly labelled `F` data or tune weights on the
same holdouts; that branch has reached its stopping condition.

The action-representation branch is complete and rejected by gameplay despite passing fresh
road and robber audits. The fixed-search road-label branch also stopped at its quality gate.
The separate outcome critic passed its offline contract, but its one authorized fixed
reranker was invalidated for hidden-information leakage. Do not add more uniform `F` rows,
tune consumed audits, move
mechanically to the third-largest imitation family, or adjust the reranker after round 8.
Only a newly frozen student-visited outcome corpus or another predeclared source of outcome
information may reopen this branch.

### 5. Keep PPO paused until a gameplay-aligned supervised branch beats the parent

PPO remains a refinement stage, not the source of a new hypothesis. The completed reward
comparison and anchor diagnostics failed to retain the hybrid-BC signal even at coefficient
`10`. Although DAgger later produced a stronger supervised parent, scaling the same PPO
dynamics without addressing their measured forgetting is not a controlled next step. Reopen
PPO only after a gameplay-aligned supervised branch beats `32031687acee`; then hold dataset, model
schema, PPO hyperparameters, seed suite, and evaluation protocol fixed and begin with a cheap
retention profile.

Use development evaluation for local checkpoint selection only. Promotion and final
decisions require their disjoint seed suites and confidence-lower-bound gates. Run one
replication for the winning treatment; treat a paired per-game outcome-score delta below 0.03 as
inconclusive.

### 6. Scale only a credible signal

The 5M `30-strong-promoted` run requires a compatible PPO checkpoint that retains at least
10% observed win rate versus `F` plus `R >= 90%`, `W >= 70%`, and `VP >= 60%` in complete
comparable reports. The retained PyTorch BC checkpoint clears those numeric values but is not
a PPO resume artifact, and prior PPO profiles failed retention. The scale run therefore
remains blocked. Preserve checkpoint, schema, manifest, environment lock, and evaluation
evidence before reversible archival.

### 7. Consider full AlphaZero only behind an evidence gate

Do not start a policy/value self-play rewrite merely because BC or PPO disappoints. Revisit
full AlphaZero-style training only if all of these are true:

- repaired MCTS supplies useful targets at a tolerable budget;
- legal/listwise BC and uniform DAgger have plateaued under locked evaluation;
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
| Factored architecture displaces MLP | Lower held-out regret and a paired gameplay interval whose lower bound clears zero |
| Road teacher merits training | Stable deterministic rankings on training-only road states, acceptable p95 latency, and a predeclared disagreement/quality gate versus `F`; run `43` failed |
| Outcome critic merits gameplay | Sufficient true outcome targets, whole-game leakage-safe splits, and better held-out win calibration plus VP-margin error than simple public-score baselines; run `47` passed |
| Critic policy-use treatment is retained | Hidden-information audit and operational intervention/latency gates pass, then a fresh paired gameplay interval strictly clears zero; run `49` was invalidated before promotion |
| Loss-conditioned DAgger is retained | Complete fresh native targets, better expanded validation/test regret, then a fresh paired gameplay interval strictly above zero; run `50` passed offline but failed gameplay decisively |
| PPO treatment wins | Paired gain at least 0.03, weak gates retained, direction repeated on one new seed |
| PPO candidate deserves 5M | F at least 10% and all weak gates retained after a controlled PPO profile; supervised strength alone does not bypass the observed PPO forgetting |
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
