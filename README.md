# Catanatron 1v1

A focused GPL-3.0 fork of [Catanatron](https://github.com/bcollazo/catanatron) for training and evaluating bots under Colonist-style 1v1 settings.

This repository is a local simulator and model-training toolkit. It does not connect to Colonist, automate play on its service, or include Catanatron's former web application.

## Scope

The retained project contains:

- a tested Catan game engine and classical search opponents;
- a two-player rules preset with a 15-point target, balanced dice, friendly robber, and a 9-card discard limit;
- a Gymnasium environment with action masks;
- schema-bound teacher-data generation and streaming behavioral cloning;
- legal-action and candidate-value losses for decision-focused imitation;
- deterministic DAgger/search-distillation data collection;
- MaskablePPO training with curriculum and checkpoint-league opponents;
- evidence-grade strength reports, artifact retention, and an optional terminal dashboard.

The upstream React UI, Flask API, database replay service, experimental package, hosted documentation, deployment files, and cloud scripts are intentionally excluded. This fork has a small CPU GitHub Actions workflow for installed-package, lint, and test checks.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/PeterLP123/catanatron-1v1.git
cd catanatron-1v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements/training-constraints.txt \
  -e ".[dev,gym,colonist,tui]"
```

Verify the 1v1 path and run a baseline match:

```bash
make test-1v1
catanatron-play --colonist-1v1 --players=F,F --num=10
```

Run a short training smoke test:

```bash
make smoke RUN_DIR=runs/smoke
```

## Training a bot

The standard workflow is teacher games, behavioral cloning, PPO, then a fixed evaluation protocol.

```bash
# 1. Generate teacher trajectories.
python examples/colonist_1v1_generate_data.py \
  --num 5000 --teachers F,F --output data/c1_ff

# 2. Pre-train the policy on legal teacher choices.
python examples/colonist_1v1_bc.py \
  --data-dir data/c1_ff --epochs 10 --loss legal_ce \
  --out runs/my_bot/bc.pt --run-dir runs/my_bot

# 3. Train with MaskablePPO and mixed league opponents.
python examples/colonist_1v1_train.py \
  --preset standard --run-dir runs/my_bot \
  --bc-checkpoint runs/my_bot/bc.pt --tensorboard

# 4. Evaluate against the milestone opponent battery.
python examples/colonist_1v1_evaluate.py \
  --agent L:runs/my_bot/colonist_maskable_ppo.zip \
  --protocol milestone --gates --eval-kind final \
  --gate-mode lower_bound \
  --report runs/my_bot/final_benchmark.json
```

See [the training guide](docs/TRAINING.md) for feature profiles, BC objectives, distillation, PPO settings, evidence suites, artifacts, and troubleshooting. Historical model numbers recorded before the 2026-07-12 evaluation-accounting fix are provisional; see [the results log](docs/RESULTS_LOG.md) before comparing checkpoints.

For NVIDIA CUDA runs, the portable wrappers under `scripts/gpu/` set up the environment,
check the device, and launch training with a dashboard in `tmux`. The
[gated GPU experiment backlog](docs/GPU_EXPERIMENT_BACKLOG.md) provides matched runs,
resource estimates, and promotion rules.

## Rules preset

The preset currently configures exactly two players, 15 victory points, the base map with official number placement, balanced dice, a friendly robber threshold based on visible victory points, and a 9-card discard limit. The implementation is documented in [the rules reference](docs/RULES.md).

Use it through the CLI or Python:

```bash
catanatron-play --colonist-1v1 --players=F,VP --num=100
```

```python
from catanatron import Color, RandomPlayer
from catanatron.colonist_1v1 import create_colonist_1v1_game

players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
game = create_colonist_1v1_game(players, seed=42)
winner = game.play()
```

## Bot codes

Player specifications are accepted by `catanatron-play` and the evaluation script.

| Code | Player |
|---|---|
| `R` | Random baseline |
| `W` | Build-weighted random baseline |
| `VP` | Immediate victory-point greedy baseline |
| `F` | Hand-crafted value-function player |
| `G:N` | Greedy player using `N` playouts per action |
| `M:N` | Monte Carlo tree search with `N` simulations |
| `AB:D` | Alpha-beta search to depth `D` |
| `L:path.zip` | MaskablePPO checkpoint |
| `T:path.pt` | Behavioral-cloning checkpoint with adjacent metadata/schema sidecars |

Run `catanatron-play --help-players` for the complete built-in list.

## Useful commands

| Command | Purpose |
|---|---|
| `make test` | Run the retained test suite |
| `make test-1v1` | Run rules, Gym, training, and evaluation tests |
| `make test-installed` | Verify imports from outside the checkout |
| `make test-gpu-ready` | Run CPU checks required before a GPU experiment |
| `make smoke` | Run the smoke training preset |
| `make train` | Run the standard preset |
| `make evaluate` | Evaluate `$(RUN_DIR)/colonist_maskable_ppo.zip` |
| `make tui` | Open the optional training dashboard |
| `python examples/colonist_1v1_backlog.py list` | Show evidence-gated experiment status |

Generated datasets, checkpoints, TensorBoard events, and run metadata belong under `data/` and `runs/`; both directories are ignored by Git. Compact validated promotion/final summaries can be published under [`docs/results/`](docs/results/README.md).

## Repository map

| Path | Responsibility |
|---|---|
| `catanatron/catanatron/` | Engine, rules adapter, players, Gym environment, training utilities |
| `examples/colonist_1v1_*.py` | Data, BC, distillation, PPO, evaluation, evidence, and TUI entry points |
| `tests/` | Engine and 1v1 regression tests |
| `docs/` | Rules, training, and architecture documentation |
| `requirements/training-constraints.txt` | Validated training compatibility envelope |
| `scripts/local_strength_eval.sh` | Reproducible local end-to-end pipeline |

For module boundaries and extension points, see [the architecture guide](docs/ARCHITECTURE.md).

## License and attribution

This fork remains licensed under GPL-3.0-or-later. Catanatron was created by Bryan Collazo and contributors; fork-specific changes are maintained in this repository. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

Colonist is a third-party product and trademark. This project is not affiliated with or endorsed by Colonist.
