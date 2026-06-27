# Colonist-style 1v1 rules reference

`catanatron.colonist_1v1.Colonist1v1Settings` is the source of truth for this fork's rules preset.

| Setting | Implemented value |
|---|---|
| Players | Exactly 2 |
| Victory target | 15 points |
| Map | `BASE` |
| Number placement | `official_spiral` |
| Dice | `balanced` |
| Friendly robber | Enabled |
| Robber protection threshold | 2 visible victory points |
| Discard limit | 9 resource cards |

The balanced dice controller starts with the 36 physical outcomes from two dice, draws without replacement, and reshuffles when fewer than 13 cards remain. It also reduces weights for recently rolled totals and adjusts seven weights using recent streaks and per-player seven counts. This is intentionally lower-variance behavior, not ordinary independent dice.

The friendly robber filters robber moves that would target a protected opponent while another legal target exists. Protection uses visible victory points, not hidden development-card points.

## Enable the preset

CLI:

```bash
catanatron-play --colonist-1v1 --players=F,VP --num=100
```

Python game:

```python
from catanatron import Color, RandomPlayer
from catanatron.colonist_1v1 import create_colonist_1v1_game

players = [RandomPlayer(Color.BLUE), RandomPlayer(Color.RED)]
game = create_colonist_1v1_game(players, seed=42)
game.play()
```

Gymnasium:

```python
import gymnasium as gym
import catanatron.gym

env = gym.make(
    "catanatron/Catanatron-v0",
    config={"colonist_1v1": True, "representation": "vector"},
)
observation, info = env.reset(seed=42)
```

The Gym learning player is player 0 (`Color.BLUE`). Supply one opponent through `config["enemies"]` or the self-play wrapper.

## Accuracy boundary

“Colonist-style” means the simulator applies the settings above to Catanatron's game engine. It is not a claim that every behavior of the current Colonist service has been independently verified. External rules can change, and this project has no service integration.

When rule parity matters:

1. compare a small set of deterministic game situations against the current external rules;
2. update `Colonist1v1Settings` or engine behavior;
3. add a regression test under `tests/test_colonist_1v1.py` or `tests/test_game.py`;
4. rerun `make test-1v1` and retrain models affected by the rule change.

Rule changes can invalidate datasets, checkpoints, and benchmark comparisons. Record the Git commit alongside any result you intend to compare later.
