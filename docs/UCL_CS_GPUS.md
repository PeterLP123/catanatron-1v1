# Running Catanatron on UCL Computer Science GPU hosts

This guide is for the shared GPU machines documented by the [UCL Computer Science Technical Support Group](https://tsg.cs.ucl.ac.uk/gpus/). It is a different service from the central UCL Myriad cluster.

## Which machine to use

| Facility | Hardware | Use for this project |
|---|---|---|
| lab105 | 25 × RTX 3090 hosts, 24 GB VRAM each | Best first choice for smoke tests, standard training and batched search experiments |
| lab121 | 30 × RTX 4060 Ti-class hosts, 16 GB VRAM each | Good fallback; ample for the current policy/value networks |
| Blaze | 4 × Titan X, 12 GB VRAM each | Dedicated longer jobs, but much older GPUs; use one card only |
| Myriad | V100 and A100 scheduled nodes | Best for repeatable jobs longer than an overnight/weekend lab window |

Prefer an RTX 3090. It has the largest memory and strongest single-GPU throughput in the CS student pool. Blaze's four GPUs do not combine into one 48 GB device, and the current code has no multi-GPU learner.

The supplied `scripts/ucl_cs/` wrappers target the single-GPU lab hosts. On Blaze, first use TSG's `/usr/local/cuda/CUDA_VISIBILITY.csh` procedure so the shared server assigns one card fairly; the training wrapper preserves an existing `CUDA_VISIBLE_DEVICES` value.

## 1. Connect

Use your **CS username**, not necessarily your central UCL username.

From the CS network, connect directly to a lab host:

```bash
ssh YOUR_CS_USERNAME@canada-l.cs.ucl.ac.uk
```

From outside CS, students connect through `knuckles`; staff use `tails`:

```bash
# Student example
ssh -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@canada-l.cs.ucl.ac.uk

# Staff example
ssh -J YOUR_CS_USERNAME@tails.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@canada-l.cs.ucl.ac.uk
```

The full lab105 and lab121 hostname lists are maintained on the [TSG GPU page](https://tsg.cs.ucl.ac.uk/gpus/). Hosts ending in `-l` boot Linux.

## 2. Find a free RTX 3090

The lab hosts do not provide a batch queue. Check a candidate host manually:

```bash
cd ~/catanatron-1v1
bash scripts/ucl_cs/gpu_check.sh
```

If it reports an existing compute process, disconnect and try another lab105 hostname. Do not start a second training job merely because some VRAM remains.

TSG notes that lab PCs are shared and may reboot at any time. They are regularly rebooted on Monday and Thursday evenings between 19:30 and midnight. Long lab jobs are safest overnight on other days or over a weekend, with frequent checkpoints.

## 3. Clone and install

The UCL helper scripts in your current Mac working tree are not on the remote machine until you copy or publish them. To use the exact local tree without making a Git commit, run this from a **Mac terminal**:

```bash
rsync -az \
  --exclude '.git/' --exclude 'venv/' --exclude 'runs/' \
  -e 'ssh -J pprender@knuckles.cs.ucl.ac.uk' \
  /Users/peterprendergast/Documents/github/catanatron-1v1/ \
  pprender@crested-l.cs.ucl.ac.uk:~/catanatron-1v1/
```

Because your CS home is shared, the copied repository will be visible from the other CS lab machines too. Rerun the same command after local code changes; it transfers only differences.

Alternatively, after these changes have been committed and pushed, clone the repository on the selected CS Linux host:

```bash
git clone https://github.com/PeterLP123/catanatron-1v1.git ~/catanatron-1v1
```

Then install on the CS Linux host:

```bash
cd ~/catanatron-1v1
bash scripts/ucl_cs/setup_env.sh
```

The setup script:

1. Finds Python 3.11 or newer, including installations under `/opt/Python`.
2. Creates `$HOME/.venvs/catanatron-1v1`.
3. Installs PyTorch 2.5.1 with CUDA 11.8.
4. Installs the project and its training/test dependencies.
5. Reports whether PyTorch can see the GPU.

If it cannot find Python 3.11, inspect the currently supported installations:

```bash
find /opt/Python -maxdepth 3 -type f -name 'python3*' 2>/dev/null
```

Then rerun with an explicit binary:

```bash
PYTHON=/path/to/python3.11 bash scripts/ucl_cs/setup_env.sh
```

If CS only exposes Python 3.10, ask TSG which user-space Python 3.11 installation they support rather than weakening this project's declared runtime requirement.

## 4. Run the smoke test

First confirm that the selected GPU is still idle:

```bash
bash scripts/ucl_cs/gpu_check.sh
```

Then run the tests and the 20,000-step smoke preset:

```bash
source "$HOME/.venvs/catanatron-1v1/bin/activate"
python -m pytest \
  tests/test_colonist_1v1.py \
  tests/test_colonist_1v1_training.py \
  tests/test_colonist_1v1_gym_training.py

TRAIN_PRESET=smoke RUN_NAME=ucl_cs_smoke \
  bash scripts/ucl_cs/train.sh
```

Artifacts are written under `runs/ucl_cs_smoke/`.

## 5. Start training with the visual command center

The recommended launcher creates one persistent `tmux` session with three windows:

1. **training** — the live Python/SB3 console output;
2. **dashboard** — a full-screen terminal UI with progress, phase, warnings, evaluations and model rankings;
3. **gpu** — `nvidia-smi`, refreshed every three seconds.

Start a standard run:

```bash
cd ~/catanatron-1v1
TRAIN_PRESET=standard RUN_NAME=ucl_cs_standard_v1 \
  bash scripts/ucl_cs/start_run.sh
```

The launcher checks that no compute process is using the GPU before it starts. It opens on the dashboard window. Inside `tmux`:

```text
Ctrl-B then N       next window
Ctrl-B then P       previous window
Ctrl-B then 0/1/2   select training/dashboard/gpu
Ctrl-B then D       detach without stopping training
```

The dashboard refreshes every two seconds. Its own shortcuts include `3` for Monitor, `4` for Ranking, `6` for Logs, `r` to refresh and `q` to close only the dashboard process.

Reconnect after closing SSH:

```bash
tmux attach -t catan-ucl_cs_standard_v1
```

Render a compact progress snapshot without opening the full-screen interface:

```bash
RUN_DIR="$HOME/catanatron-1v1/runs/ucl_cs_standard_v1" SNAPSHOT=1 \
  bash scripts/ucl_cs/dashboard.sh
```

## 6. Run standard training manually

Use `tmux` so an SSH disconnection does not terminate the process:

```bash
tmux new -s catan
cd ~/catanatron-1v1
TRAIN_PRESET=standard RUN_NAME=ucl_cs_standard_v1 \
  bash scripts/ucl_cs/train.sh
```

Detach with `Ctrl-B`, then `D`. Reconnect later with:

```bash
tmux attach -t catan
```

Check progress from another SSH session:

```bash
tail -f ~/catanatron-1v1/runs/ucl_cs_standard_v1/console.log
nvidia-smi
```

To resume from a saved PPO checkpoint after an interruption:

```bash
RESUME_CHECKPOINT="$HOME/catanatron-1v1/runs/ucl_cs_standard_v1/checkpoints/ppo_colonist_50000_steps.zip" \
TRAIN_PRESET=standard RUN_NAME=ucl_cs_standard_v1_resume \
  bash scripts/ucl_cs/train.sh
```

The current standard run should fit comfortably on both 16 GB and 24 GB hosts. Do not run the `strong` or `overnight` presets on a lab PC until checkpoint intervals and host reboot risk have been reviewed.

## 7. Data and evaluation

Teacher-game generation and evaluation are mainly CPU work. They do not justify holding a shared GPU. Run them locally, on a non-GPU CS machine, or through the Myriad CPU scripts.

If you must run a short evaluation on the GPU host, stop training first and execute:

```bash
source "$HOME/.venvs/catanatron-1v1/bin/activate"
python examples/colonist_1v1_evaluate.py \
  --agent L:runs/ucl_cs_standard_v1/colonist_maskable_ppo.zip \
  --protocol fast \
  --num-games 200 \
  --gates \
  --report runs/ucl_cs_standard_v1/evaluation.json
```

## Storage and failure handling

- Store code, manifests and checkpoints in your CS home or an approved CS project store, not only on a lab PC's local disk.
- TSG states that project stores are backed up monthly, incrementally backed up nightly, and snapshotted weekly.
- Copy important run directories back to your Mac after each milestone.
- Lab machines may reboot without notice; a `tmux` session does not survive a reboot.
- The current standard preset saves every 50,000 timesteps. For future long search/self-play jobs, checkpoint every 10–15 minutes.

The approximately 10 GiB currently free in your shared home is enough for one Python environment, a smoke run and probably one standard run when pip caching is disabled. The setup script now disables pip's download cache automatically. Request more storage before accumulating datasets, parallel experiment runs or search replay buffers.

Recommended storage targets:

| Stage | Working space | Why |
|---|---:|---|
| Smoke + one standard run | 10 GiB free | One environment and limited checkpoints |
| Several ablations | 30–50 GiB free | Separate run directories and TensorBoard logs |
| Search training | 100 GiB+ project store | Replay buffers, decision labels and historical champions |

For the strong-bot research phase, ask TSG or your supervisor for an approved CS project store rather than treating the lab host's local disk as durable storage. TSG lists `request@cs.ucl.ac.uk` for support; project stores are backed up and snapshotted.

Copy a result home from macOS:

```bash
rsync -az -e 'ssh -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk' \
  YOUR_CS_USERNAME@canada-l.cs.ucl.ac.uk:~/catanatron-1v1/runs/ucl_cs_standard_v1/ \
  ./runs/ucl_cs_standard_v1/
```

Use `tails.cs.ucl.ac.uk` instead of `knuckles.cs.ucl.ac.uk` if you are staff.

## Recommendation

Use the CS service in this order:

1. RTX 3090 lab105 host for environment setup, smoke tests and short experiments.
2. RTX 4060 Ti lab121 host when no 3090 is free.
3. Myriad V100 for scheduled, reproducible long runs.
4. Myriad A100 only after the batched search learner demonstrably saturates a V100.

This keeps development fast without relying on an unattended lab PC for multi-day training.
