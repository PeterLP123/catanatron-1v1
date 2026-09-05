# Run 58: 128 versus 32 search simulations

Specification fixed on 2026-09-04 before gameplay evaluation.

Hypothesis: the retained run-55 search needs more simulations to improve on its
frozen policy and could then supply useful targets for policy distillation.

## Fixed treatment

Change only `num_simulations` from 32 to 128 in the run-55 manifest. Keep the
run-32 policy, checkpoint and sidecar hashes, public-F leaf, sqrt(2) PUCT
constant, action boundary, policy fallback, and final-move rule fixed. Do not
enable run 57's own-hand bonus or run 56's chance extension. No source-code
change is needed for this comparison.

Run directory: `runs/58-public-f-puct-128-s20260904`.

## Evaluation fixed before results

1. Validate both manifests and run the existing visible-search regression tests.
   Verify that `num_simulations` is their only different field.
2. Run the existing 20-game operational diagnostic for each budget on the same
   fresh seeds, base seed `2026090458`, sequentially to avoid local CPU contention.
   Require zero errors, truncations, forbidden expansions or opponent-turn
   expansions; at least 200 searches and 20 multi-ply decisions; p95 search
   latency at most 100 ms. Record policy changes as a diagnostic, with bounds
   explicitly set to 0–100%; making more changes is not a success criterion.
3. Candidate and run 55 each play 100 balanced-seat games against F on the same
   development schedule, round 59. Separately play 100 balanced-seat direct games
   against run 55 on development round 60.
4. Proceed only if both operational diagnostics pass, all pilot games finish
   without errors or truncations, the paired F win-rate gain is at least three
   percentage points, VP-margin gain is nonnegative, and direct score against
   run 55 is at least 50%. A pilot pass is not a promotion.
5. If eligible, compare candidate and run 55 on a fresh 400-game-per-bot schedule
   against F, promotion round 18. Require the paired 95% bootstrap lower bound
   to be strictly above zero. Then run the final round-18 fast battery, 50 games
   per opponent, requiring R/W/VP point floors of 90/70/60%, with no errors or
   truncations. Report the separate absolute F target of 52%.

Use one thread per numeric library and the existing builder, diagnostic and
evaluation CLIs. Preserve exact commands, immutable manifests, source hashes,
package versions, complete game rows and decisions. Audit proposed game seeds
against historical local reports before starting. No tuning, budget sweep,
schedule extension or reuse after viewing results.

## Conditional next step

Only a confirmed stronger search teacher proceeds to a separately frozen,
bounded distillation pilot: collect legal-action root visit probabilities,
train on the full distribution with replay of existing examples, and compare
the resulting policy plus 32-simulation search against run 55 on fresh games.
Fix that training specification before collecting data or fitting weights.

If the search-budget experiment fails a gate, stop this branch and retain run
55. Do not generate a search-labelled training corpus. Opponent-turn search
with explicit hidden-state uncertainty remains a larger follow-up project.

## Outcome

Completed on 2026-09-04. The 128-simulation treatment **missed the development
gate** and is not retained. Run 55 remains the playable bot. Confirmation, the
final battery and distillation were not run.

| Test | 128 simulations | Run 55: 32 simulations | Decision |
|---|---|---|---|
| Operational, 20 games each | 864 searches, 641 multi-ply choices; all gates passed | 850 searches, 612 multi-ply choices; all gates passed | Proceed |
| Search latency, mean / p95 | 34.79 / 82.03 ms | 14.04 / 35.88 ms | Both below the 100 ms p95 limit |
| Policy overrides / genuine choices | 11 / 2,053 (0.54%) | 9 / 2,051 (0.44%) | Diagnostic only |
| Development vs F, 100 games each | 41 wins, VP margin -1.36 | 39 wins, VP margin -1.86 | Paired +2 points, 95% CI [-3, +8]; misses the fixed +3-point gate |
| Direct match, 100 games | 53 wins, VP margin +0.31 | 47 wins | Passes the direct-score gate |

The paired F schedules contained 36 shared wins, 56 shared losses, five parent
losses changed to candidate wins, and three reversals. Direct first/second-seat
win rates were 52%/54%; the overall Wilson interval was [43.29%, 62.49%].
Operational timings come from sequential runs on matching seeds; trajectories
can diverge, so they are not measurements of identical individual decisions.

All 340 evaluation games completed without errors or truncations. Source and
manifest hashes remained fixed, and the paired bootstrap was reproduced from
the full reports. All 13 existing visible-search regression tests and Ruff
passed. No new tests, wrappers or search-code changes were needed for this
parameter comparison. The result does not establish a stronger teacher, so no
search-labelled corpus was generated and no model weights were trained.

Promotion and final round 18 remain unconsumed. Do not tune or extend the
consumed development rounds 59/60. This is a failed fixed-budget pilot, not
evidence that every larger search budget is ineffective.

## Reproduction evidence

The run directory preserves the original pre-results `specification.md`,
`commands.json`, `experiment_lock.json`, source patch and package snapshot,
complete operational and per-game reports, execution logs, `pilot_decision.json`
and `decision.json`. The schedule audit checked 235 historical JSON files and
14,110 game rows, with no collisions or parsing failures.

- Candidate manifest: `b7efc26e78be00a5d6533b39aea03fe4acafe3ac8bdaec732bdae5ea7576b365`.
- Retained run-55 manifest: `36e4707e76215605f4ba55334f074e3afda4ee7d3d2c8f8cfb89b03a5ac5c3f3`.
- Unchanged policy checkpoint: `32031687acee2636e60a2b2c9bf667580f7a884e5e223ddc230d131726381943`.
- Source-set SHA-256: `f141db939f9f2a77a39be26a1efc6f5ef3c6512e917ddda668e58292d6c14103`.
- Frozen pre-results specification: `ad3c091f968299cccb65e31f5b4b6527199dc498c8dfd81cf0202e95ddb8ef18`.
- Paired development comparison: `1b16dabbddf05faf8f283ba014ba37a65994af1eecf9eacb6bf91552ca0e6845`.
- Final decision: `295fe753bc2f50026a90a53fe46b461cbab7f53d9e48326f2b4aee0a66b1df83`.

These are development results. No compact promotion/final artifact is published
under `docs/results/` because neither locked evaluation stage was reached.
