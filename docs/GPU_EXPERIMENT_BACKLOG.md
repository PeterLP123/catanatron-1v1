# GPU experiment backlog

This is the ordered run queue for the UCL RTX 3090 Ti hosts. It is intentionally gated: the results log already shows four approaches at roughly 0–1% versus `F`, so another long PPO run is not justified until a cheap experiment produces a real early signal.

The executable definitions live in `catanatron.gym.experiment_backlog`. Every training entry launches through the existing `tmux` command center and writes to `runs/<experiment-id>/`.

## Use the queue

After syncing the repository and installing the UCL environment:

```bash
cd ~/catanatron-1v1
source "$HOME/.venvs/catanatron-1v1/bin/activate"

python examples/colonist_1v1_backlog.py list
python examples/colonist_1v1_backlog.py next
python examples/colonist_1v1_backlog.py show 10-balanced-actual-s101
```

Launch the next experiment in the visual dashboard:

```bash
python examples/colonist_1v1_backlog.py start 00-gpu-smoke
```

The launcher enforces completed dependencies. `--force` exists for results produced in another directory, but should not be the normal workflow.

## Ordered training queue

The estimates are deliberately ranges. A previous 500k run took about 17 minutes and occupied roughly 320 MiB on the development Mac; the UCL host may be CPU-bound despite its faster GPU. Use the first standard run to replace these estimates with measured UCL wall time and disk usage.

| ID | Question | Expected GPU time | Run storage | Promotion rule |
|---|---|---:|---:|---|
| `00-gpu-smoke` | Does CUDA, checkpointing, evaluation and the dashboard work end to end? | 0.1–0.3 h | 0.15 GiB | Must finish 20k steps without CUDA errors |
| `10-balanced-actual-s101` | Control: balanced curriculum with current actual-VP shaping | 0.3–1.0 h | 0.4 GiB | Record the two-seat scorecard |
| `11-balanced-visible-s101` | Does public-score-only shaping beat the stronger actual-VP learning signal? | 0.3–1.0 h | 0.4 GiB | ≥0.03 weighted-score gain; no R/W/VP regression |
| `12-balanced-actual-s202` | Does an actual-VP advantage repeat under seed 202? | 0.3–1.0 h | 0.4 GiB | Run only if actual VP wins seed 101 |
| `13-balanced-visible-s202` | Does a visible-VP advantage repeat under seed 202? | 0.3–1.0 h | 0.4 GiB | Run only if visible VP wins seed 101 |
| `20-hard-bc-actual-s101` | Does choice-focused BC improve the decision margin before PPO? | 0.4–1.2 h | 0.4 GiB | Lower held-out regret and improve F rate or VP margin |
| `21-hard-bc-visible-s101` | Does hard-state BC combine better with visible shaping? | 0.4–1.2 h | 0.4 GiB | Run only if visible reward won the first A/B |
| `30-strong-promoted` | Does the best credible 500k model benefit from 5M strong-curriculum steps? | 3–10 h | 3.5 GiB | Start only after ≥10% vs F with all weak gates retained |
| `40-selfplay-polish` | Can short self-play improve a strong anchored model without collapse? | 0.4–1.5 h | 0.5 GiB | Search/F rises; R/W/VP stay within two points |

Do not automatically run both seed-202 jobs. Run the seed-202 version of the reward treatment that wins seed 101. If the seed-101 difference is smaller than 0.03 weighted score, treat the result as inconclusive and defer replication until there is a better hypothesis.

## Compare matched runs

Once both reward runs finish:

```bash
python examples/colonist_1v1_backlog.py compare \
  10-balanced-actual-s101 \
  11-balanced-visible-s101
```

Reports produced after the 2026-07-01 seat-order correction use genuinely fixed first/second seats and deterministic game schedules. Do not compare their per-seat fields with older reports; re-evaluate an old checkpoint under the new protocol first.

The comparison reports weighted score, R/W/VP/F win rates, largest seat gap and aggregate gate status from each `final_benchmark.json`.

Use a milestone evaluation before promoting a winner:

```bash
WINNER=10-balanced-actual-s101
python examples/colonist_1v1_evaluate.py \
  --agent "L:runs/$WINNER/colonist_maskable_ppo.zip" \
  --protocol milestone --num-games 200 --gates \
  --report "runs/$WINNER/milestone_200.json"
```

## Prepare the hard-state branch

This is primarily CPU work and can start as soon as the larger project store is mounted. Candidate scoring is intentionally slower because it evaluates every legal action.

```bash
mkdir -p data/hard_state_v2 runs/hard-state-bc

python examples/colonist_1v1_generate_data.py \
  --num 2000 --teachers F,F --score-candidates --seed 101 \
  --output data/hard_state_v2/F_F

python examples/colonist_1v1_generate_data.py \
  --num 2000 --teachers VP,F --score-candidates --seed 101 \
  --output data/hard_state_v2/VP_F
```

Before committing GPU time, run a small schema test and inspect the generated metadata:

```bash
python -m pytest \
  tests/machine_learning/test_dataset_v2.py \
  tests/machine_learning/test_leaf_evaluation.py

du -sh data/hard_state_v2
find data/hard_state_v2 -maxdepth 2 -name dataset_meta.json -print
```

Train the choice-focused BC checkpoint on the GPU:

```bash
python examples/colonist_1v1_bc.py \
  --data-dir data/hard_state_v2/F_F data/hard_state_v2/VP_F \
  --epochs 10 --hard-states --test-fraction 0.1 --split-seed 101 \
  --tensorboard runs/hard-state-bc/tb \
  --out runs/hard-state-bc/bc.pt \
  --run-dir runs/hard-state-bc
```

The BC gate is not raw action accuracy. Inspect held-out `legal_choice_accuracy` and `mean_regret`, then evaluate the checkpoint in both seats:

```bash
python examples/colonist_1v1_evaluate.py \
  --agent T:runs/hard-state-bc/bc.pt \
  --opponent F --num-games 200 \
  --report runs/hard-state-bc/vs_F_200.json
```

Only launch the PPO warm-start when the new BC checkpoint improves held-out regret:

```bash
python examples/colonist_1v1_backlog.py start \
  20-hard-bc-actual-s101 \
  --bc-checkpoint "$PWD/runs/hard-state-bc/bc.pt"
```

## Promotion commands

If a 500k candidate reaches at least 10% versus `F` while retaining the weak gates:

```bash
python examples/colonist_1v1_backlog.py start \
  30-strong-promoted \
  --resume-checkpoint "$PWD/runs/20-hard-bc-actual-s101/colonist_maskable_ppo.zip"
```

After the strong run passes a full evaluation, the optional self-play polish is:

```bash
python examples/colonist_1v1_backlog.py start \
  40-selfplay-polish \
  --resume-checkpoint "$PWD/runs/30-strong-promoted/colonist_maskable_ppo.zip"
```

## Tests to run before GPU access

These are CPU-only and catch failures that would otherwise waste a shared GPU session:

```bash
make test-gpu-ready

python examples/colonist_1v1_env_benchmark.py \
  --n-envs 4 --steps 500 --report runs/environment_benchmark.json

python examples/colonist_1v1_search_benchmark.py \
  --budgets 10,25,50,100 --profile-only
```

Then run the complete 1v1 suite:

```bash
make test-1v1
```

## Storage plan

- The four cheap 500k profiles together should need about 2 GiB of run storage, excluding the Python environment.
- Data now writes atomic 100-game Parquet shards. A measured 20-game scored probe was about 90% smaller when sharded; budget roughly 0.2–0.5 GiB for the first 4,000-game hard-state dataset, then replace the estimate with `du -sh`.
- Reserve roughly 4 GiB for a promoted 5M run with rolling checkpoints.
- Keep at least 20% headroom so a checkpoint write cannot fail midway.

A 30–50 GiB allocation is ample for this initial queue if old checkpoints are pruned. A 100 GiB+ project store remains useful for many historical search/self-play generations, but is no longer required for the first dataset.

## Stop rules

1. Stop a run on NaNs, repeated CUDA errors, a full disk, or no progress events for 15 minutes.
2. Do not promote a 500k candidate merely because its weak-opponent score improves; it must show a material `F` signal or better F VP margin.
3. Do not start the 5M profile if all 500k candidates remain in the existing 0–4% F band.
4. Never infer a winner from one seed when the weighted-score difference is below 0.03.
5. Preserve the winning final checkpoint, manifest, evaluation report and model registry before pruning a run directory.
