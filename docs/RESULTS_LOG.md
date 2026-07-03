# Results log

Two-seat scorecards for each run, so local and cloud runs stay comparable across commits.
Compare only within the same protocol / rules. Win rates are **two-seat** (the agent plays
`num_games/2` as the first seat and `num_games/2` as the second) unless noted. `seat[s0/s1]`
is the per-seat win rate; a large gap flags a one-seat specialist (not human-like; see the
[current execution plan](PLAN.md#current-execution-plan)).

> **Seat-order correction — 2026-07-01.** The evaluator previously reversed the supplied
> player list, but `State` shuffled that list again when each game started. Historical aggregate
> win rates still represent mixed-seat performance, but historical `seat0/seat1` splits are not
> trustworthy. New evaluations preserve explicit order and use fixed seed schedules; re-run a
> checkpoint before making a seat-balance decision.

---

## 500k baseline — `runs/ec2_proxy_500k` (Phase 0 reference)

- **Date:** 2026-06-28
- **Eval commit:** 0c00e81 + uncommitted Phase 0 two-seat eval changes
- **Agent:** `L:runs/ec2_proxy_500k/colonist_maskable_ppo.zip` (MaskablePPO, 500k steps, BC warm-start)
- **Protocol:** `full`, 200 games/opponent, two-seat (`both_seats=True`)
- **Report:** `runs/ec2_proxy_500k/eval_two_seat_no_m200.json`
- **Deferred to cloud:** `G:25` (~90s+/game) and `M:200` (~2min/game) — too slow locally;
  see the [GPU experiment backlog](GPU_EXPERIMENT_BACKLOG.md). `AB:2` is the only search bot
  fast enough to run locally.

| Opponent | Gate | Win rate | seat0 / seat1 | VP diff | Result |
|---|---|---|---|---|---|
| R     | 90% | 92.9% | 88% / 98% | +9.91  | PASS |
| W     | 70% | 88.5% | 80% / 97% | +8.77  | PASS |
| VP    | 60% | 83.3% | 76% / 91% | +7.49  | PASS |
| F     | 52% | **0.5%**  | 0% / 1%  | −10.80 | **FAIL** |
| AB:2  | 52% | **0.0%**  | 0% / 0%  | −10.76 | **FAIL** |
| G:25  | 52% | (deferred — cloud) | | | — |
| M:200 | 52% | (deferred — cloud) | | | — |

Weighted score 0.396; 3/5 local gates passed.

**Findings**

- **Crushed by the value-function bots.** 0.5% vs `F` and 0% vs `AB:2` (which shares F's
  hand-crafted value function), losing by ~11 VP in short ~85-turn games. This established the
  decision-margin problem: the self-play–leaning 500k run never learned to beat real heuristic play.
- **Seat imbalance on the weak baselines.** The model is ~10–17 pts *stronger* in seat 1
  (second player) than seat 0 across R/W/VP. The old first-seat-only numbers (R 88 / W 78 /
  VP 78) essentially reported the model's *weaker* seat — so the prior measurement understated
  the weak-baseline win rates while the F result (0) was unaffected. Seat balance becomes a
  first-class gate in the [current plan](PLAN.md#decision-gates).

---

## Phase 1 — the F wall (negative results, 2026-06-29)

Phase 1 set out to beat `F` (ValueFunctionPlayer) via diverse-teacher BC → PPO. **Four
independent methods all failed at ~0% vs F**, which produced the current gated strategy in
[PLAN.md](PLAN.md). All numbers two-seat.

| Method | Artifact | vs F | vs R / W / VP | Note |
|---|---|---|---|---|
| Self-play PPO 500k | `runs/ec2_proxy_500k` | 0.5% | 93 / 89 / 83 | Phase 0 baseline above |
| BC, 5.4M samples (F,F + VP,F) | `runs/v2/bc.pt` | 0.5% | 81 / 55 / 67 | val action-acc 75.6%; imitation ceiling |
| PPO 500k trained *directly* vs F | `runs/v2_ppo_fheavy` | ~1% | 97 / 85 / 86 | F win-rate **flat 0–4%** across all 50k evals; VP margin flat at −11 |
| Model-based: 1-ply lookahead + **learned** value | `runs/v3/value.pt` | 0.5% | **37** / – / – | value net 99.85% win-acc, yet barely beats Random |

**Why (the decisive finding):** the model-based agent is *mechanically identical* to F —
same `game.copy()/execute()` 1-ply lookahead — differing **only** in the value function.
It still loses to F and can't reliably beat Random (games drag to ~520 turns). So the wall
is **value-function quality at the decision margin**, not architecture, not reactive-vs-
lookahead, not scale: a value net trained on terminal outcomes is accurate globally
(late-game states dominate and are obvious) but ~uninformative in the early/mid game where
real choices happen, so 1-ply argmax over it is near-random. F's hand-tuned heuristic gives
a meaningful gradient (city > road > pass) at every move.

(`G:25` and `M:200` were not run here — too slow locally, see the cost note above. The env
plays to **15 VP**; `colonist_1v1=True` overrides evaluate_matchup's `vps_to_win`.)

---

## Best available human-like reactive bot (partial deliverable, 2026-06-29)

After the F wall, the target became a **human-like reactive bot** that convincingly beats the
weak/random tiers, plays naturally (no superhuman lookahead), and is competitive-but-not-dominant
vs the hand-crafted lookahead bots. That complete target is not yet met. The best current
candidate is **`runs/v2_ppo_fheavy/colonist_maskable_ppo.zip`**: it is strongest against R and
the **most seat-balanced** of our agents (the 500k baseline had 10–17pt seat gaps; this one is
within ~9pt), a side-benefit of training heavily against F even though it never beat F.

- **Protocol:** 200 games/opponent, two-seat. R/W/VP/F/AB:2 (G:25, M:200 deferred — cloud).

| Opponent | Win rate | seat0 / seat1 | VP diff | Tier |
|---|---|---|---|---|
| R    | 97.4% | 97% / 98% | +10.40 | weak — **dominant** |
| W    | 85.3% | 81% / 90% | +7.90  | weak — **strong** |
| VP   | 85.6% | 81% / 90% | +7.86  | weak — **strong** |
| F    | 1.0%  | 1% / 1%   | −10.10 | lookahead — not competitive |
| AB:2 | 0.5%  | 0% / 1%   | −10.50 | lookahead — not competitive |

This meets the weak-tier portion of the target with good seat balance. The full deliverable
remains open because F/AB are not yet competitive; they remain the stretch track below.

### Stretch track — beating F (future, out of local scope)

The evidence says beating F requires a value/policy that is well-shaped **at the decision
margin**, which naive outcome-regression and 500k model-free RL do not provide. The credible
approach is **AlphaZero-style**: a policy+value net trained by iterated self-play + MCTS,
with **value targets bootstrapped from search** (not raw game outcomes), likely needing cloud
compute. This is a multi-session research effort with uncertain payoff on this hardware and is
intentionally deferred. The corrected two-seat evaluation and this results log are the
comparison harness for that work when/if it starts.
