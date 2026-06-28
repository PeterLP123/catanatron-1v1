# Results log

Two-seat scorecards for each run, so local and cloud runs stay comparable across commits.
Compare only within the same protocol / rules. Win rates are **two-seat** (the agent plays
`num_games/2` as the first seat and `num_games/2` as the second) unless noted. `seat[s0/s1]`
is the per-seat win rate; a large gap flags a one-seat specialist (not human-like — see
[docs/PLAN.md](PLAN.md) Phase 3).

---

## 500k baseline — `runs/ec2_proxy_500k` (Phase 0 reference)

- **Date:** 2026-06-28
- **Eval commit:** 0c00e81 + uncommitted Phase 0 two-seat eval changes
- **Agent:** `L:runs/ec2_proxy_500k/colonist_maskable_ppo.zip` (MaskablePPO, 500k steps, BC warm-start)
- **Protocol:** `full`, 200 games/opponent, two-seat (`both_seats=True`)
- **Report:** `runs/ec2_proxy_500k/eval_two_seat_no_m200.json`
- **Deferred to cloud:** `G:25` (~90s+/game) and `M:200` (~2min/game) — too slow locally;
  see Phase 2. `AB:2` is the only search bot fast enough to run locally.

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
  hand-crafted value function), losing by ~11 VP in short ~85-turn games. This is the Phase 1
  problem to solve: the self-play–leaning 500k run never learned to beat real heuristic play.
- **Seat imbalance on the weak baselines.** The model is ~10–17 pts *stronger* in seat 1
  (second player) than seat 0 across R/W/VP. The old first-seat-only numbers (R 88 / W 78 /
  VP 78) essentially reported the model's *weaker* seat — so the prior measurement understated
  the weak-baseline win rates while the F result (0) was unaffected. Seat balance becomes a
  first-class gate in Phase 3.
