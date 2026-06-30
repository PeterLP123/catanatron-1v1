import math
import time
from collections import defaultdict
import random

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron.players.leaf_evaluation import (
    leaf_win_probability,
    make_position_value_fn,
    state_signature,
)
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions

SIMULATIONS = 10
epsilon = 1e-8
EXP_C = 2**0.5


def _as_bool(value):
    """Coerce CLI strings to bool. ``bool("False")`` is True, so parse explicitly."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class MCTSPlayer(Player):
    def __init__(
        self,
        color,
        num_simulations=SIMULATIONS,
        prunning=False,
        value_fn_name="base_fn",
        max_time_ms=None,
        use_leaf_cache=True,
    ):
        super().__init__(color)
        self.num_simulations = int(num_simulations)
        self.prunning = _as_bool(prunning)
        # Only store picklable config; the F value function is a closure and is
        # built per-decision so the player stays picklable (it can be shipped to
        # multiprocessing workers by other players in the same game).
        self.value_fn_name = value_fn_name
        # Optional wall-clock budget per decision (ms). When set, search runs
        # until the budget elapses instead of a fixed simulation count, so
        # strength can be measured at a fixed latency budget. CLI passes strings.
        self.max_time_ms = (
            float(max_time_ms) if max_time_ms not in (None, "", "None") else None
        )
        # Transposition cache of F leaf evaluations, keyed by state signature.
        # Values are exact functions of the signature, so cached entries are
        # always valid; it persists across decisions for a higher hit rate.
        self.use_leaf_cache = _as_bool(use_leaf_cache)
        self._leaf_cache = {} if self.use_leaf_cache else None
        self._leaf_cache_cap = 200_000
        # Profiler output from the most recent decide() (simulations, nodes/sec).
        self.last_search_stats = None

    def __getstate__(self):
        # Don't ship the (potentially large) cache to pickling workers; it is a
        # pure optimization and is rebuilt empty on the other side.
        state = self.__dict__.copy()
        state["_leaf_cache"] = {} if self.use_leaf_cache else None
        return state

    def _leaf_value_fn(self, stats=None):
        # Build the positional F value function once per decision and reuse it as
        # the search leaf evaluator (replaces random playouts) across all leaves.
        pos_value_fn = make_position_value_fn(self.value_fn_name)
        cache = self._leaf_cache
        cap = self._leaf_cache_cap

        def evaluate(game, color):
            if cache is None:
                return leaf_win_probability(game, color, pos_value_fn)
            key = state_signature(game, color)
            cached = cache.get(key)
            if cached is not None:
                if stats is not None:
                    stats["leaf_cache_hits"] += 1
                return cached
            value = leaf_win_probability(game, color, pos_value_fn)
            if len(cache) < cap:
                cache[key] = value
            if stats is not None:
                stats["leaf_cache_misses"] += 1
            return value

        return evaluate

    def decide(self, game: Game, playable_actions):
        actions = list_prunned_actions(game) if self.prunning else playable_actions
        if len(actions) == 1:
            return actions[0]

        stats = {
            "expansions": 0,
            "leaf_evals": 0,
            "leaf_cache_hits": 0,
            "leaf_cache_misses": 0,
        }
        root = StateNode(
            self.color,
            game.copy(),
            None,
            self.prunning,
            self._leaf_value_fn(stats),
            stats,
        )

        budget_s = self.max_time_ms / 1000.0 if self.max_time_ms else None
        start = time.perf_counter()
        simulations = 0
        while True:
            root.run_simulation()
            simulations += 1
            if budget_s is None:
                if simulations >= self.num_simulations:
                    break
            elif (time.perf_counter() - start) >= budget_s:
                break
        elapsed = time.perf_counter() - start

        lookups = stats["leaf_cache_hits"] + stats["leaf_cache_misses"]
        self.last_search_stats = {
            "simulations": simulations,
            "expansions": stats["expansions"],
            "leaf_evals": stats["leaf_evals"],
            "leaf_cache_hits": stats["leaf_cache_hits"],
            "leaf_cache_misses": stats["leaf_cache_misses"],
            "leaf_cache_hit_rate": (
                stats["leaf_cache_hits"] / lookups if lookups else 0.0
            ),
            "elapsed_s": elapsed,
            "simulations_per_s": simulations / elapsed if elapsed > 0 else float("inf"),
            "nodes_per_s": (
                stats["expansions"] / elapsed if elapsed > 0 else float("inf")
            ),
        }
        return root.best_action_by_visits()

    def __repr__(self):
        return super().__repr__() + f"({self.num_simulations}:{self.prunning})"


class StateNode:
    def __init__(
        self,
        color,
        game: Game,
        parent,
        prunning=False,
        leaf_value_fn=None,
        stats=None,
    ):
        self.level = 0 if parent is None else parent.level + 1
        self.color = color  # color of player carrying out MCTS
        self.parent = parent
        self.game = game  # state
        self.children = []
        self.prunning = prunning
        # Heuristic leaf evaluator, shared down the tree. Falls back to the F
        # win-probability leaf so standalone StateNodes still evaluate correctly.
        self.leaf_value_fn = leaf_value_fn or (lambda g, c: leaf_win_probability(g, c))
        # Optional shared profiler counters (expansions, leaf_evals).
        self.stats = stats

        self.wins = 0
        self.visits = 0
        self.result = None  # set if terminal

    def run_simulation(self):
        # select
        tmp = self
        tmp.visits += 1
        while not tmp.is_leaf():
            tmp = tmp.select()
            tmp.visits += 1

        if not tmp.is_terminal():
            # expand
            tmp.expand()
            tmp = tmp.select()
            tmp.visits += 1

            # evaluate the leaf with the F value function (not a random playout)
            value = tmp.leaf_value()
        else:
            # Read the winner from the *selected terminal leaf*, not the root.
            winner = tmp.game.winning_color()
            value = 1.0 if winner == self.color else 0.0

        if self.stats is not None:
            self.stats["leaf_evals"] += 1

        # backpropagate
        tmp.backpropagate(value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        return self.game.winning_color() is not None

    def expand(self):
        children = defaultdict(list)
        playable_actions = self.game.playable_actions
        actions = list_prunned_actions(self.game) if self.prunning else playable_actions
        for action in actions:
            outcomes = execute_spectrum(self.game, action)
            for state, proba in outcomes:
                children[action].append(
                    (
                        StateNode(
                            self.color,
                            state,
                            self,
                            self.prunning,
                            self.leaf_value_fn,
                            self.stats,
                        ),
                        proba,
                    )
                )
        self.children = children
        if self.stats is not None:
            self.stats["expansions"] += 1

    def select(self):
        """Descend into a child StateNode for the in-tree (UCT) policy."""
        action = self._select_action(exploration=True)

        # Sample the chance outcome for the chosen action by its probability.
        children = self.children[action]
        children_states = list(map(lambda c: c[0], children))
        children_probas = list(map(lambda c: c[1], children))
        return random.choices(children_states, weights=children_probas, k=1)[0]

    def _select_action(self, exploration=True):
        """Pick an action among the *expanded* ones using a negamax UCT score.

        Scores are kept from the MCTS player's perspective (``self.color``).
        When it is the opponent's turn at this node, they minimize our win
        probability, so the stored value is flipped before ranking.
        """
        actions = list(self.children.keys())
        scores = [self._action_score(a, exploration) for a in actions]
        idx = max(range(len(scores)), key=lambda i: scores[i])
        return actions[idx]

    def best_action_by_visits(self):
        """Final move choice: the most-visited (robust) expanded action."""
        actions = list(self.children.keys())
        visits = [sum(child.visits for child, _ in self.children[a]) for a in actions]
        idx = max(range(len(visits)), key=lambda i: visits[i])
        return actions[idx]

    def _action_score(self, action, exploration=True):
        to_move = self.game.state.current_color()
        opponent_turn = to_move != self.color
        score = 0.0
        for child, proba in self.children[action]:
            if child.visits == 0:
                q = 0.5  # neutral prior for an unvisited child
            else:
                q = child.wins / child.visits  # self.color win-rate proxy in [0, 1]
            if opponent_turn:
                q = 1.0 - q  # opponent maximizes their own win probability
            u = (
                EXP_C
                * (math.log(self.visits + epsilon) / (child.visits + epsilon)) ** 0.5
                if exploration
                else 0.0
            )
            score += proba * (q + u)
        return score

    def leaf_value(self):
        """Heuristic value of this leaf in ``[0, 1]`` from ``self.color``'s view."""
        return self.leaf_value_fn(self.game, self.color)

    def backpropagate(self, value):
        self.wins += value

        tmp = self
        while tmp.parent is not None:
            tmp = tmp.parent
            tmp.wins += value
