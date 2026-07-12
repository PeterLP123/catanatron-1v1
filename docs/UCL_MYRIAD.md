# Running Catanatron 1v1 on UCL Myriad

This is the central UCL Research Computing service. If your access is through the Computer Science student GPU hosts documented at `tsg.cs.ucl.ac.uk/gpus`, use [the CS GPU host guide](UCL_CS_GPUS.md) first.

Myriad is the appropriate UCL service for this project: it is designed for high-throughput, single-node and GPU jobs. Standard use is free at the point of use. The current GPU pool includes V100 nodes and 40 GB or 80 GB A100 nodes.

The repository includes an Apptainer image definition and Grid Engine job scripts under `scripts/ucl/`. These validate and run the existing training pipeline. They do not by themselves implement the search-supervised bot described in the strong-bot plan.

## 1. Get access

Apply at <https://signup.rc.ucl.ac.uk/>. Every successful Research Computing application receives Myriad access.

- Students and postdoctoral researchers need approval from a permanent UCL staff member, normally their supervisor or PI. The form asks for the sponsor's UCL username.
- Permanent staff accounts are approved automatically.
- UCL says account creation can take up to one working day after approval.
- Support: `rc-support@ucl.ac.uk`.

## 2. Connect from macOS

From the UCL network or while connected to the UCL VPN:

```bash
ssh YOUR_UCL_ID@myriad.rc.ucl.ac.uk
```

From outside the UCL network, UCL requires either the VPN or the SSH gateway. The gateway requires SSH-key setup for external access. Once configured:

```bash
ssh -J YOUR_UCL_ID@ssh-gateway.ucl.ac.uk \
  YOUR_UCL_ID@myriad.rc.ucl.ac.uk
```

An optional local `~/.ssh/config` entry makes this shorter:

```sshconfig
Host ucl-gateway
  HostName ssh-gateway.ucl.ac.uk
  User YOUR_UCL_ID

Host myriad
  HostName myriad.rc.ucl.ac.uk
  User YOUR_UCL_ID
  ProxyJump ucl-gateway
  ServerAliveInterval 60
```

Then connect with `ssh myriad`. Do not copy a private SSH key into this repository or onto shared project storage.

## 3. Put the repository in Scratch

On Myriad:

```bash
mkdir -p "$HOME/Scratch"
cd "$HOME/Scratch"
git clone https://github.com/PeterLP123/catanatron-1v1.git
cd catanatron-1v1
gquota
```

Myriad currently gives users a 1 TB default quota. Home and Scratch are working storage, not durable archival storage; copy important checkpoints elsewhere.

## 4. Build the reproducible runtime

UCL's centrally installed PyTorch is older than this project requires. The supplied Apptainer definition uses Python 3.11 and PyTorch 2.5.1 with CUDA 11.8, which supports both Myriad's V100 and A100 generations.

Build once on a Myriad login node:

```bash
cd "$HOME/Scratch/catanatron-1v1"
bash scripts/ucl/build_image.sh
```

The result is `$HOME/Scratch/catanatron-1v1.sif`. The image is several gigabytes. To rebuild it after dependency changes:

```bash
FORCE=1 bash scripts/ucl/build_image.sh
```

The project source is bind-mounted at job time, so ordinary Python edits do not require an image rebuild. Rebuild only when dependencies or the container definition change.

## 5. Validate the GPU setup

Start with a V100. The current networks fit comfortably in 16 GB, and there are more V100 nodes than A100 nodes. From the repository root on Myriad:

```bash
qsub -ac allow=EF scripts/ucl/gpu_smoke.qsub
qstat
```

The job checks `nvidia-smi`, verifies `torch.cuda.is_available()`, runs the 1v1 tests, and executes the 20,000-step smoke preset. Its output directory is:

```text
runs/ucl_smoke_<JOB_ID>/
```

Grid Engine also writes the combined shell log beside the submitted job script because the script uses `-cwd` and `-j y`.

Useful scheduler commands:

```bash
qstat                  # queued and running jobs
qstat -j JOB_ID        # job details
qexplain JOB_ID        # explain an Eqw scheduling error
qdel JOB_ID            # cancel a job
```

## 6. Run a measured training job

Run the standard 500,000-step preset on a V100:

```bash
qsub -ac allow=EF \
  -v TRAIN_PRESET=standard,RUN_NAME=ucl_standard_v1 \
  scripts/ucl/train.qsub
```

To warm-start from a repository-relative BC checkpoint:

```bash
qsub -ac allow=EF \
  -v TRAIN_PRESET=standard,RUN_NAME=ucl_standard_bc,BC_CHECKPOINT_REL=runs/v2/bc.pt \
  scripts/ucl/train.qsub
```

Only request an A100 after profiling shows a GPU bottleneck:

```bash
qsub -ac allow=L \
  -v TRAIN_PRESET=strong,RUN_NAME=ucl_strong_a100 \
  scripts/ucl/train.qsub
```

The PPO script now defaults to `SubprocVecEnv` whenever a preset uses more than one environment. It shares curriculum progress across workers and reloads the file-backed league index as checkpoints are promoted. Run `examples/colonist_1v1_env_benchmark.py` on the allocated node before changing CPU requests; process overhead can make small worker counts slower even when larger counts improve throughput.

## 7. Generate teacher data as CPU array jobs

Do not spend GPU queue time on simulation-only data generation. Ten default tasks produce 1,000 `F,F` games total:

```bash
qsub scripts/ucl/generate_data.qsub
```

Generate 2,000 `VP,F` games as twenty independent tasks:

```bash
qsub -t 1-20 \
  -v GAMES_PER_TASK=100,DATASET_NAME=c1_vpf,TEACHER_A=VP,TEACHER_B=F,BASE_SEED=0 \
  scripts/ucl/generate_data.qsub
```

Array task `i` starts at `BASE_SEED + (i - 1) × GAMES_PER_TASK`, preventing duplicate trajectories across tasks. Each task writes an atomic 100-game Parquet shard.

Each task writes to node-local `$TMPDIR` and copies one completed shard back to:

```text
data/<DATASET_NAME>/part_<TASK_ID>/
```

This avoids generating thousands of small Parquet writes directly on Myriad's shared filesystem.

## 8. Evaluate without occupying a GPU

Evaluation is primarily simulator-bound. Submit it to a CPU node:

```bash
qsub -v MODEL_REL=runs/ucl_standard_v1/colonist_maskable_ppo.zip,PROTOCOL=fast,EVAL_GAMES=200 \
  scripts/ucl/evaluate.qsub
```

The report is written next to the model under `ucl_eval_<JOB_ID>/evaluation.json`.
The supplied scheduler wrapper currently produces a manual point-gate diagnostic report.
It is useful for cluster operation checks but is not publishable promotion/final evidence.
For a model decision, run the current evaluation CLI with `--eval-kind final` and
`--gate-mode lower_bound`, preserve the protocol count, and publish the validated report as
described in [the training guide](TRAINING.md#6-publish-evidence-and-retain-artifacts).

## Resource choices

| Workload | Queue request | Reason |
|---|---|---|
| Tests and search correctness | CPU node | No neural batching benefit yet |
| Teacher/search data generation | CPU array jobs | Scale across many serial simulations |
| Current PPO smoke/standard | 1 V100, 8 CPU slots | 16 GB is ample; validates CUDA cheaply |
| Batched policy/value self-play | 1 V100 initially | Upgrade only after measuring saturation |
| Large batches or graph model | 1 A100 40 GB | Use `-ac allow=L` when 16 GB is insufficient |
| Multi-GPU | Not yet | Current code has no distributed learner |

## Operational rules

- Never run training or data generation on a login node. UCL permits only short, non-intensive tests there.
- Keep jobs at or below Myriad's 48-hour limit for requests using 2–36 cores. Checkpoint long training runs so they can resume.
- Run `gquota` before large data jobs. A full quota can prevent both results and scheduler log files from being written.
- Keep authoritative checkpoints outside cluster scratch using RDSS, OneDrive/rclone, or a local copy.
- Record the Git commit, container image checksum, job ID, node type, `nvidia-smi` output, and evaluation seeds with every promoted model.

Copy a completed run back to the local machine, using the VPN or configured gateway:

```bash
rsync -az myriad:~/Scratch/catanatron-1v1/runs/ucl_standard_v1/ \
  ./runs/ucl_standard_v1/
```

## Official UCL references

- [Myriad accounts, nodes, GPU requests and limits](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/)
- [Account application process](https://www.rc.ucl.ac.uk/docs/Account_Services/)
- [Remote login and file transfer](https://www.rc.ucl.ac.uk/docs/howto/)
- [Grid Engine example jobs](https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/)
- [Apptainer on UCL clusters](https://www.rc.ucl.ac.uk/docs/Software_Guides/Singularity/)
- [Myriad status](https://www.rc.ucl.ac.uk/docs/Status_page/)
