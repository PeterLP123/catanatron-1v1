# Run 57: own-hand construction readiness

Specification fixed on 2026-09-04 before gameplay evaluation.

Hypothesis: the retained run-55 search can make better spending decisions when its
leaf evaluator distinguishes useful resource combinations in its own hand.

## Fixed treatment

Keep run 55's policy and critic references/hashes, 32 simulations, sqrt(2) PUCT
constant, action boundary, tie-breaking, and policy fallback. The only manifest
change is `leaf_evaluator=public_f_own_hand_v1`.

For each eligible city or settlement, reserve the resource cost, then count how
many missing cards can be obtained by trading surplus cards at owned 2:1 or 3:1
ports (otherwise 4:1). Readiness is `1 - remaining_missing_cards / total_cost`.
Use the maximum readiness over eligible builds, never their sum. A city requires
an owned settlement and an available city piece; a settlement requires a connected
legal location and an available piece. Initial placement receives zero bonus.

Add `0.5 * 0.6 * readiness` to the existing public-F win-probability logit. Thus
the bonus is at most half the existing public-VP coefficient. Terminal values
remain exactly zero or one. Opponent resource identities, hidden development
cards, deck order, and future random outcomes are never inputs. Bank shortages
and future production are not modelled by this affordability estimate.

This is one fixed heuristic, with no coefficient or formula tuning on the pilot.
Run 55 remains the default unless the confirmation gates below pass.

## Evaluation fixed before results

Run directory: `runs/57-own-hand-public-f-s20260904`.

1. Unit/information-boundary checks: known-hand sensitivity, cost reservation,
   ports, impossible builds, terminal scores, and invariant values/actions after
   changing an opponent's hidden composition while preserving public counts.
2. Existing 20-game operational diagnostic, base seed `2026090457`: zero errors,
   truncations, forbidden expansions or opponent-turn expansions; at least 200
   searches and 20 multi-ply choices; policy change rate 1–50%; p95 below 100 ms.
3. Candidate and run 55 each play 100 games against `F`, on the same balanced-seat
   development schedule, round 57. Separately play 100 balanced-seat direct games
   against run 55 on development round 58.
4. Proceed to confirmation only if the diagnostic passes, all pilot games are
   accounted for without errors or truncations, the paired F win-rate gain is at
   least 3 percentage points, VP-margin gain is nonnegative, and the direct score
   against run 55 is at least 50%. A pilot pass is not a promotion.
5. If eligible, run a fresh 400-game-per-bot paired comparison against F on
   promotion round 17. Retain only if the paired 95% bootstrap lower bound is
   strictly above zero and the final round-17 fast battery (50 games/opponent)
   keeps R/W/VP floors of 90/70/60%. Report the separate absolute F target of 52%.

Use existing builder, diagnostic, paired-evaluator and single-opponent evaluator
CLIs. Save exact argv, source hashes, immutable manifests, complete per-game
reports and comparisons in the run directory. Verify new game schedules do not
overlap historical reports before starting. Stop on integrity/operational failure;
do not extend consumed schedules or tune after viewing results.

## Outcome

Completed on 2026-09-04. The candidate is **not retained**; run 55 remains the
playable bot. The larger confirmation did not establish an improvement.

| Test | Result | Decision |
|---|---|---|
| Operational, 20 games | 717 searches, 510 multi-ply choices, 1.12% policy changes, 18.02 ms p95; all nine gates passed | Proceed |
| Development vs F, 100 games per bot | Candidate 38%, run 55 35%; paired +3 points, 95% CI [0, +7]; VP margin +0.42 | Clears the predeclared pilot point-estimate gate |
| Direct vs run 55, 100 games | 54 wins, 46 losses; VP margin +0.28 | Clears the pilot head-to-head gate |
| Fresh confirmation vs F, 400 games per bot | Candidate 120/400 (30%), run 55 119/400 (29.75%); paired +0.25 points, 95% CI [-1.75, +2.25]; VP margin -0.0325 | Fails confirmation; final battery skipped |

All 1,120 evaluation games completed without errors or truncations. The 464-test
regression suite, Ruff and Black passed. The candidate remains an opt-in
experimental evaluator for reproducing this result; no coefficient or formula was
tuned after the pilot. This result concerns the fixed v1 treatment, not every
possible way of using own-hand information.

The frozen pre-results specification is preserved at
`runs/57-own-hand-public-f-s20260904/specification.md`, SHA-256
`a9c52af6111903f1e99111ae41758523436031c116c381773339d276aa3678b4`.
That directory also contains exact commands, a source patch, package versions,
full game rows, operational diagnostics, and the immutable final decision.

- Candidate manifest: `e2fb8e57e2114f09a6a3064fee756fe8cba84c39650b7576fe14eeb5a3e6550b`.
- Retained run-55 manifest: `36e4707e76215605f4ba55334f074e3afda4ee7d3d2c8f8cfb89b03a5ac5c3f3`.
- Unchanged policy checkpoint: `32031687acee2636e60a2b2c9bf667580f7a884e5e223ddc230d131726381943`.
- Source-set SHA-256: `f141db939f9f2a77a39be26a1efc6f5ef3c6512e917ddda668e58292d6c14103`.
- Paired confirmation: `d53cb47f688948c938d31ae2d2e754d7b48e57e53cd4382ab293a7699c028205`.

Validated compact confirmation reports: [candidate](../results/57-own-hand-puct-r17-candidate.json)
and [run-55 control](../results/57-own-hand-puct-r17-parent.json). Their `rejected`
status also reflects the separate absolute F win-rate gate, which neither passes.
