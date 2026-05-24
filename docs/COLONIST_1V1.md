# Colonist.io 1v1 — training guide

This guide covers the **end-to-end workflow** in this repository: simulating Colonist-style 1v1 games, generating teacher data, behavioral cloning (BC), Maskable PPO training with a checkpoint league, and evaluating strength against fixed baselines.

For a short overview, see the [README](../README.md).

---

## Table of contents

1. [Rules and environment](#rules-and-environment)
2. [Dependencies](#dependencies)
3. [Workflow](#workflow)
4. [Step 1 — Teacher data](#step-1--teacher-data)
5. [Step 2 — Behavioral cloning](#step-2--behavioral-cloning)
6. [Step 3 — PPO training](#step-3--ppo-training)
7. [Step 4 — Evaluation](#step-4--evaluation)
8. [Training presets and curriculum](#training-presets-and-curriculum)
9. [Rewards](#rewards)
10. [Run directory reference](#run-directory-reference)
11. [CLI and learned players](#cli-and-learned-players)
12. [TUI command center](#tui-command-center)
13. [Troubleshooting](#troubleshooting)
14. [Python API](#python-api)

---

## Rules and environment

Colonist 1v1 is enabled with `colonist_1v1=True` on `catanatron/Catanatron-v0` or `--colonist-1v1` on `catanatron-play`.

Settings are defined in `catanatron.colonist_1v1.Colonist1v1Settings`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num_players` | 2 | Head-to-head only |
| `vps_to_win` | 15 | Match Colonist 1v1 win condition |
| `dice_mode` | `balanced` | Fairer dice distribution |
| `friendly_robber` | `True` | Robber restrictions apply |
| `friendly_robber_vp_threshold` | 2 | Uses **visible** VP for robber targeting |
| `discard_limit` | 9 | Hand size before forced discard |
| `map_type` | `BASE` | Standard board |
| `number_placement` | `official_spiral` | Official number spiral |

The learning agent is always **player 0** (first color in the env). Opponents are supplied via `config["enemies"]`, a league, or `SelfPlayEnv` with an `opponent_factory`.

---

## Dependencies

**Core (editable install):**

```bash
pip install -e ".[gym,dev]"
```

**Colonist ML stack (install explicitly — not pinned in `pyproject.toml`):**

```bash
pip install torch stable-baselines3 sb3-contrib tensorboard pyarrow
```

**Optional:**

```bash
pip install -e ".[tui]"    # Textual training TUI
```

| Package | Used for |
|---------|----------|
| `gymnasium`, `numpy`, `pandas`, `fastparquet` | Env + parquet (via `[gym]`) |
| `torch` | BC and checkpoint loading |
| `stable-baselines3`, `sb3-contrib` | MaskablePPO + action masking |
| `pyarrow` | Parquet teacher logs |
| `tensorboard` | `--tensorboard` on train script |
| `textual` | `colonist_1v1_tui.py` |

---

## Workflow

Recommended order:

1. **Generate** teacher trajectories → `data/<run>/` parquet + `dataset_meta.json`
2. **BC** → `bc.pt` + `bc.meta.json`
3. **PPO** → `colonist_maskable_ppo.zip`, `checkpoints/`, `league/`
4. **Evaluate** → `final_benchmark.json` (and optional mid-training `eval_reports/`)

Each training run should use its own `--run-dir` (e.g. `runs/c1_strong`) so manifests and leagues do not collide.

---

## Step 1 — Teacher data

**Script:** `examples/colonist_1v1_generate_data.py`

Wraps `catanatron-play` with `--colonist-1v1` and Parquet accumulation.

```bash
python examples/colonist_1v1_generate_data.py --num 5000 --teachers F,F --output data/c1_ff
python examples/colonist_1v1_generate_data.py --num 2000 --teachers VP,F --output data/c1_vpf
```

| Flag | Description |
|------|-------------|
| `--num` | Number of games |
| `--output` | Directory for `*.parquet` |
| `--teachers` | Two player codes, e.g. `F,F` or `VP,F` |
| `--include-board-tensor` | Slower; adds board tensor columns |

**Tips:**

- Prefer **`F`**, **`VP`**, or **`G:N`** over **`W,W`** — weak teachers produce weak BC labels.
- A `.colonist_run_started` marker is written so BC only loads parquet from the current generation batch when multiple runs share a folder.
- `dataset_meta.json` records `num_games` for consistent file selection.

Equivalent CLI:

```bash
catanatron-play --colonist-1v1 --players=F,F --num 5000 \
  --output data/c1_ff --output-format parquet
```

---

## Step 2 — Behavioral cloning

**Script:** `examples/colonist_1v1_bc.py`

Trains a small MLP to predict teacher actions from the vector observation.

```bash
python examples/colonist_1v1_bc.py \
  --data-dir data/c1_ff \
  --epochs 10 \
  --batch-size 4096 \
  --out runs/my_run/bc.pt
```

Outputs:

- `bc.pt` — PyTorch `state_dict` for the MLP
- `bc.meta.json` — `obs_dim`, `n_actions`, `hidden_sizes`, `val_accuracy`, etc.

| Flag | Default | Notes |
|------|---------|--------|
| `--data-dir` | (required) | One or more parquet directories |
| `--epochs` | 5 | Increase for larger datasets |
| `--n-actions` | 332 | Full discrete action space size |
| `--hidden` | 512 512 | Must match PPO `--hidden` for warm-start |
| `--run-dir` | — | Writes `TrainingRunTracker` events |

BC weights are copied into MaskablePPO’s policy MLP during PPO via `warmstart_bc_into_maskable_ppo`.

---

## Step 3 — PPO training

**Script:** `examples/colonist_1v1_train.py`

Main orchestrator: vectorized envs, checkpointing, optional mid-run eval, league registration, BC warm-start, mixed opponents.

### Presets

| Preset | Timesteps | `n_envs` | Eval every | Curriculum |
|--------|-----------|----------|------------|------------|
| `smoke` | 20k | 1 | 10k | balanced |
| `standard` | 500k | 4 | 50k | balanced |
| `strong` | 5M | 8 | 250k | strong |
| `overnight` | 20M | 8 | 500k | strong |

Presets also set `save_freq`, `eval_games`, enable `--mixed-league`, and pick a curriculum. Override with `--preset custom` and explicit flags.

### Example commands

```bash
# Quick sanity check
python examples/colonist_1v1_train.py --preset smoke --run-dir runs/smoke --skip-final-eval

# Standard run after BC
python examples/colonist_1v1_train.py --preset standard --run-dir runs/c1_std \
  --bc-checkpoint runs/c1_std/bc.pt --tensorboard

# Strong run (long; use GPU if available)
python examples/colonist_1v1_train.py --preset strong --run-dir runs/c1_strong \
  --bc-checkpoint runs/c1_strong/bc.pt --mixed-league --tensorboard \
  --final-eval-protocol milestone
```

### Important flags

| Flag | Purpose |
|------|---------|
| `--run-dir` | All artifacts (checkpoints, league, manifest) |
| `--bc-checkpoint` | Warm-start PPO from BC |
| `--resume-checkpoint` | Continue from an SB3 zip |
| `--n-envs` | Parallel `DummyVecEnv` workers |
| `--mixed-league` | Sample league / teacher / baseline each episode |
| `--curriculum` | `none`, `balanced`, `strong`, `self_play` |
| `--league-size` | Max checkpoints kept in `league/` |
| `--eval-freq` | Mid-training benchmark (0 = off) |
| `--eval-protocol` | `fast`, `milestone`, `full` for mid-run eval |
| `--final-eval-protocol` | Protocol for post-training report |
| `--skip-final-eval` | Faster iteration |
| `--visible-vp-reward` | Shape reward on public VP only |
| `--hidden H1 H2` | Policy MLP width (default 512 512) |
| `--league-checkpoints` | Seed league with existing zips |

### What happens during training

1. **Env** — `colonist_1v1=True`, vector observations, `colonist_shaped_reward` (or visible-VP variant).
2. **Action masking** — `ActionMasker` + `MaskablePPO` (invalid actions never sampled).
3. **Checkpoints** — Saved under `checkpoints/ppo_colonist_<steps>_steps.zip`.
4. **League** — New checkpoints copied to `league/`; old entries pruned to `--league-size`.
5. **Mixed opponents** — With `--mixed-league`, each reset may pick a league snapshot, a classical teacher (`F`, `VP`, …), or weak `W`, with weights from the active curriculum stage.
6. **Promotions** — Mid-eval can copy `best_fast` / `best_f` into `league/promoted/`.
7. **Final model** — `colonist_maskable_ppo.zip` at end of `learn()`.

---

## Step 4 — Evaluation

**Scripts:**

- `examples/colonist_1v1_evaluate.py` — Single agent vs one opponent or full battery
- `examples/colonist_1v1_benchmark_report.py` — Aggregate / compare reports

```bash
# Full battery with gates
python examples/colonist_1v1_evaluate.py \
  --agent L:runs/my_run/colonist_maskable_ppo.zip \
  --benchmark --gates --protocol full \
  --report runs/my_run/eval_full.json

# Quick check vs ValueFunction only
python examples/colonist_1v1_evaluate.py --agent L:runs/my_run/colonist_maskable_ppo.zip \
  --opponent F --num-games 200
```

### Eval protocols

| Protocol | Opponents | Games (default) | Use case |
|----------|-----------|-----------------|----------|
| `fast` | R, W, VP, F | 50 | Frequent training feedback |
| `milestone` | + G:25 | 100 | Promotion decisions |
| `full` | R, W, VP, F, G:25, M:200, AB:2 | 200 | Publication-grade strength |

### Default win-rate gates

Applied with `--gates` (agent is player 0):

| Opponent | Min win rate |
|----------|----------------|
| R | 90% |
| W | 70% |
| VP | 60% |
| F | 52% |
| G:25 | 52% |
| M:200 | 52% |
| AB:2 | 52% |

Reports include **Wilson score intervals**, a **weighted_score** (see `DEFAULT_SCORE_WEIGHTS` in `colonist_1v1_eval.py`), and `all_gates_passed`.

**Note:** Beating `F` (value-function bot) is hard; a policy that crushes `R`/`W`/`VP` but loses to `F` is common early in training. Use curriculum and stronger teacher data before expecting to pass the `F` gate.

---

## Training presets and curriculum

### Curriculum stages

Curricula are piecewise schedules keyed on **environment steps** (`balanced`, `strong`, `self_play`). Each stage sets:

- `league_weight` — Sample a checkpoint from the league
- `teacher_weight` — Sample a classical bot (`teacher_codes`)
- `baseline_weight` — Sample weak bot (usually `W`)

Example (`strong` preset): early training favors `VP` and `F` teachers; later stages increase league weight and may include `G:25` playouts.

Override teacher list for all stages:

```bash
python examples/colonist_1v1_train.py --preset standard --curriculum balanced \
  --teacher-codes F VP --mixed-league --run-dir runs/custom_teachers
```

### Checkpoint league

`CheckpointLeague` (`catanatron.gym.colonist_training`) maintains `league/index.json` and copies zips into `league/`. `SelfPlayEnv` loads `L:<path>` opponents via `Sb3CheckpointPlayer`.

With `--mixed-league`, new checkpoints registered during the **same** PPO run can appear as opponents without restarting training.

---

## Rewards

Default: `colonist_shaped_reward` in `catanatron.gym.colonist_rewards`.

- **Terminal:** +1 win, −1 loss for the learning player (`p0_color`).
- **Shaping:** `0.02 × ΔVP` since that player’s last turn (actual VP by default).

Public-only shaping (closer to what humans see on Colonist):

```bash
python examples/colonist_1v1_train.py --visible-vp-reward ...
```

Or in code: `make_colonist_shaped_reward(use_visible_vp=True)`.

---

## Run directory reference

Example layout for `runs/my_run/`:

```
runs/my_run/
├── run_manifest.json          # run_id, phases, hyperparameters
├── training_events.jsonl      # structured log (eval, promotion, …)
├── models_index.jsonl         # eval history rows
├── bc.pt / bc.meta.json       # if BC run in same dir
├── colonist_maskable_ppo.zip  # final SB3 policy
├── checkpoints/
│   └── ppo_colonist_*_steps.zip
├── league/
│   ├── index.json
│   ├── ppo_colonist_*.zip
│   └── promoted/              # best_fast, best_f copies
├── eval_reports/              # mid-training eval_step_*.json
├── final_benchmark.json
└── tb/                        # TensorBoard (if enabled)
```

---

## CLI and learned players

### Colonist 1v1 simulations

```bash
catanatron-play --colonist-1v1 --players=F,F --num 1000 --output data/logs --output-format parquet
```

### Play learned bot vs baseline

```bash
catanatron-play --colonist-1v1 --players=L:runs/my_run/colonist_maskable_ppo.zip,F --num 50
catanatron-play --colonist-1v1 --players=T:runs/my_run/bc.pt,F --num 50
```

Player codes are listed in the [README](../README.md#bot-player-codes-catanatron-play). Implementation: `catanatron.cli.cli_players`.

---

## TUI command center

```bash
pip install -e ".[tui]"
python examples/colonist_1v1_tui.py
```

Features:

- Discover runs under `runs/`
- Build shell commands for data / BC / train / eval
- Live job runner and registry sparklines

Non-interactive snapshot (CI / scripts):

```bash
python examples/colonist_1v1_tui.py --run-dir runs/my_run --once
```

---

## Troubleshooting

| Symptom | Likely cause | What to try |
|---------|----------------|-------------|
| BC loss flat / low val accuracy | Weak teachers or too few games | Regenerate with `F,F` or `VP,F`; increase `--num` |
| PPO wins vs R/W but 0% vs F | Policy overfits weak pool | `--preset strong`, `--curriculum strong`, more teacher weight early |
| `FileNotFoundError` on parquet | Empty or wrong `--data-dir` | Check `dataset_meta.json` and `.colonist_run_started` |
| MaskablePPO errors on load | Action space / hidden size mismatch | Keep `--hidden` and `--n-actions` consistent between BC and PPO |
| Slow eval | `full` protocol with MCTS/AB | Use `--protocol fast` during training; `full` only at milestones |
| OOM with many envs | Large vector obs + 8 envs | Reduce `--n-envs` or use smaller preset |

### Tests

```bash
pytest tests/test_colonist_1v1.py tests/test_colonist_1v1_training.py tests/test_colonist_1v1_gym_training.py -q
```

---

## Python API

### Create a Colonist 1v1 game

```python
from catanatron import RandomPlayer, Color
from catanatron.colonist_1v1 import create_colonist_1v1_game

players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
game = create_colonist_1v1_game(players, seed=42)
winner = game.play()
```

### Gym env for training

```python
import gymnasium as gym
import catanatron.gym
from catanatron.gym.colonist_rewards import colonist_shaped_reward
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron import Color

env = gym.make(
    "catanatron/Catanatron-v0",
    config={
        "colonist_1v1": True,
        "representation": "vector",
        "enemies": [WeightedRandomPlayer(Color.RED)],
        "reward_function": colonist_shaped_reward,
    },
)
```

### Run benchmark from Python

```python
from catanatron.colonist_1v1_eval import run_benchmark, DEFAULT_BENCHMARK_GATES

report = run_benchmark(
    "L:runs/my_run/colonist_maskable_ppo.zip",
    protocol="milestone",
    gates=DEFAULT_BENCHMARK_GATES,
)
print(report.summary, report.all_gates_passed)
report.write_json("report.json")
```

---

## Related files

| Path | Role |
|------|------|
| `catanatron/catanatron/colonist_1v1.py` | Rules + `create_colonist_1v1_game` |
| `catanatron/catanatron/colonist_1v1_eval.py` | Protocols, gates, reports |
| `catanatron/catanatron/gym/colonist_training.py` | League, BC utils, presets, tracker |
| `catanatron/catanatron/gym/colonist_rewards.py` | Shaped reward |
| `catanatron/catanatron/gym/wrappers/self_play.py` | League / factory opponents |
| `examples/colonist_1v1_*.py` | CLI entry points |
