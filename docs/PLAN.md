# Plan: toward a robust, human-like 1v1 Catan bot

## Context

The strongest existing model (`runs/ec2_proxy_500k`, 500k PPO steps) beats the weak
baselines (R 88%, W 78%, VP 78%) but **loses 0/100 to `F`**, the hand-crafted value bot,
averaging just 3.9 VP. Three root causes, established by reading the run artifacts and code:

1. **No BC warm-start** — that run has no `bc.*` artifacts; PPO learned from scratch.
2. **Self-play–leaning opponent mix** — its `league/` holds only its own checkpoints; the
   policy converged to beat clones of itself and collapses against real heuristic play.
3. **Biased measurement** — `evaluate_matchup` (`colonist_1v1_eval.py:204-214`) always seats
   the agent first, so every win rate is inflated by first-player advantage and per-seat
   robustness is invisible.

**Chosen goal:** robust, *human-like* 1v1 play — treat the simulator as a proxy for real
Colonist, avoid overfitting to one heuristic, and require strength in *both* seats.
**Compute:** hybrid — iterate locally at the `standard` (500k) preset, promote milestones to
`strong` (5M) cloud bursts. The EC2/cloud scripts were removed in the refocus, so a
lightweight cloud path must be restored.

Beating `F` and the built-in battery is the shared prerequisite for *any* use case; the
human-like goal then drives the measurement, anti-overfitting, and robustness emphasis below.

## Phase 0 — Trustworthy two-seat measurement (do first; cheap)

Nothing else can be judged until measurement is unbiased. This is pure eval-side work.

- **Add seat-swapped evaluation** in `catanatron/catanatron/colonist_1v1_eval.py`:
  - Extend `evaluate_matchup` (`:204-275`) with a `both_seats: bool = True` option that splits
    `num_games` between agent-as-seat-0 and agent-as-seat-1, reusing the existing per-game loop
    and `_agent_color_from_players` (`:200`). Track the agent color per game rather than
    assuming index 0.
  - Add `win_rate_seat0`, `win_rate_seat1`, and per-seat VP margin to `MatchupResult`
    (`:167-176`); keep the combined `win_rate` + `wilson_score_interval` (`:136`) as today.
  - Thread the flag through `run_benchmark` (`:350-415`) and record it in `build_eval_meta`
    (`:277-301`).
- **Expose `--both-seats/--first-seat-only`** in `examples/colonist_1v1_evaluate.py` (default
  both seats).
- **Add a regression test** in `tests/test_colonist_1v1_training.py`: a deterministic agent
  that always wins is counted correctly across both seats; combined win rate = mean of seats.
- **Re-baseline the 500k model**: run the `full` protocol two-seat to get honest reference
  numbers (R/W/VP/F/G:25/M:200/AB:2, per seat). This becomes the comparison point for all
  later runs.

## Phase 1 — A bot that actually beats F (local, 500k)

- **Diverse-teacher data, not F-only.** `data/c1` is ~4,845 `F,F` games; imitating one
  heuristic risks copying its quirks. Generate additional sets with
  `examples/colonist_1v1_generate_data.py` for `VP,F` and a search teacher (`M:100` or `G:25`),
  so BC learns varied strong play.
- **BC warm-start** with `examples/colonist_1v1_bc.py --data-dir data/c1 data/c1_vp_f data/c1_search`
  → `runs/v2/bc.pt` (keep default `--hidden 512 512`, action head 332, per `bc.meta.json`).
- **BC checkpoint gate:** two-seat eval of `T:runs/v2/bc.pt` vs `F`. A faithful imitation
  should be roughly competitive (≥ ~40%); if not, fix data/epochs *before* spending PPO time.
- **PPO with diverse opponents** (`examples/colonist_1v1_train.py`):
  `--preset standard --bc-checkpoint runs/v2/bc.pt --mixed-league --curriculum balanced`.
  `balanced` (`colonist_training.py:280-292`) front-loads F/VP/W teachers via
  `make_mixed_opponent_factory` (`:338-400`) — the opposite of the collapsed self-play run.
  **Avoid the `self_play` curriculum at this stage.**
- **Reward aligned to the goal:** add `--visible-vp-reward` so shaping uses *public* VP
  (`make_colonist_shaped_reward(use_visible_vp=True)`, `colonist_rewards.py:49-81`) — a human
  can't see hidden dev-card VP, and the friendly-robber rule already keys on visible VP. Treat
  actual-VP vs visible-VP as a quick A/B.
- **Milestone target:** pass R/W/VP gates and **beat F (≥52%)** on two-seat eval. This unblocks
  everything downstream.

## Phase 2 — Robust strength via cloud bursts (5M)

- Once F is cleared locally, promote to a `strong` (5M) run with `--curriculum strong`
  (`colonist_training.py:293-317`), which escalates opponents to `F` then `G:25` — the path
  to beating the search bots (G:25, M:200, AB:2) in the `full` protocol.
- **Restore a lightweight cloud path** (removed in the refocus): add a minimal
  `scripts/cloud_train.sh` that provisions a GPU box, installs `pip install -e ".[gym,colonist]"`,
  runs the `strong` preset into a `runs/<name>`, and syncs `runs/` back. Keep it burst-sized,
  not a full platform — document it in `docs/TRAINING.md`.
- Keep `--mixed-league` so the bounded `CheckpointLeague` (`:174-247`, last-8) is sampled
  alongside teachers/baselines rather than dominating.

## Phase 3 — Robustness & anti-overfitting (the human-like core)

This is where the chosen goal diverges from "just pass the gates."

- **Held-out / out-of-distribution eval:** test against opponents *not* in the training mix —
  e.g. deeper search (`M:500`, `AB:3`) — to detect overfitting to the trained opponents.
- **Seat balance as a first-class gate:** require per-seat win rates vs F (from Phase 0) to
  *both* clear threshold and sit within a few points of each other — a one-seat specialist is
  not human-like.
- **Exploitability probe:** run the bot vs progressively deeper AB/MCTS and inspect loss-mode
  and VP-margin *distributions* (not just win rate) for repeatable losing lines.
- **Self-play polish last:** only after the bot beats heuristics, a short `self_play`-curriculum
  stage sharpens play (at high anchor strength it refines rather than collapses).
- **Investigation (scope as follow-up):** audit whether the env observation leaks hidden
  information (opponent hand / dev-card VP) in `gym/envs/`. If it does, an imperfect-information
  observation is the most principled change for human-like play — sized separately as it touches
  the observation space and would require retraining.

## Phase 4 — Reproducibility (supports hybrid iteration)

- Reuse existing tracking (`run_manifest.json`, `training_events.jsonl`, `models_index.jsonl`,
  the TUI). Add `docs/RESULTS_LOG.md` recording each run's config + two-seat scorecard so local
  and cloud runs stay comparable across commits (the docs already warn to compare same
  protocol/commit/rules).
- Pin seeds in `EVAL_PROTOCOLS` (`colonist_1v1_eval.py:54`) for repeatable scorecards.

## Critical files

| File | Change |
|---|---|
| `catanatron/catanatron/colonist_1v1_eval.py` | Seat-swapped `evaluate_matchup`, per-seat fields on `MatchupResult`, thread through `run_benchmark`/`build_eval_meta`; pin protocol seeds |
| `examples/colonist_1v1_evaluate.py` | `--both-seats` flag (default on) |
| `examples/colonist_1v1_train.py` | Use `--bc-checkpoint`, `--mixed-league`, `--curriculum balanced`→`strong`, `--visible-vp-reward` (mostly existing flags) |
| `examples/colonist_1v1_generate_data.py` | Generate `VP,F` and search-teacher datasets |
| `tests/test_colonist_1v1_training.py` | Seat-swap eval regression test |
| `scripts/cloud_train.sh` (new) | Minimal cloud-burst training path |
| `docs/RESULTS_LOG.md` (new), `docs/TRAINING.md` | Results log + cloud-burst docs |

Reused as-is: `make_mixed_opponent_factory`, `CURRICULUM_PRESETS`, `CheckpointLeague`,
`warmstart_bc_into_maskable_ppo`, `TRAINING_PRESETS` (`colonist_training.py`);
`make_colonist_shaped_reward` (`colonist_rewards.py`); `wilson_score_interval`,
`EVAL_PROTOCOLS`, `DEFAULT_BENCHMARK_GATES`, `summarize_report` (`colonist_1v1_eval.py`);
bot codes `T:` (BC) and `L:` (PPO).

## Verification

- `make test-1v1` plus the new seat-swap test pass.
- Two-seat `full` eval of the 500k baseline produces honest per-seat reference numbers.
- BC checkpoint gate: `T:runs/v2/bc.pt` vs F is competitive before PPO.
- Phase 1 local PPO run clears R/W/VP gates and beats F (≥52%) two-seat.
- Phase 2 cloud `strong` run improves the search-bot win rates vs the 500k baseline, with
  seat balance maintained.

## Notes / non-goals

- Still no connection to or automation of Colonist; this stays a local simulator + trainer.
- "Human-like" is pursued via robustness proxies (diverse opponents, both-seat strength,
  held-out eval, visible-VP shaping), since no human-game data is in scope.