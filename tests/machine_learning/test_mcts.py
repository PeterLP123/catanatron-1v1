import random

from typing import List
from catanatron import Game, RandomPlayer, Color
from catanatron.models.player import Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.mcts import MCTSPlayer, StateNode


def test_root_node_initial_properties():
    """
    Tests the initial properties of a root StateNode.
    """
    # 1. Create a real Game object
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game_instance = Game(players)

    player_color = Color.BLUE

    # 2. Create a StateNode instance for the root of a search tree
    root_node = StateNode(
        color=player_color,
        game=game_instance,
        parent=None,  # A root node has no parent
        prunning=False,  # Default or explicit
    )

    # 3. Assert initial properties
    assert root_node.wins == 0, "Initial wins should be 0"
    assert root_node.visits == 0, "Initial visits should be 0"
    assert root_node.is_leaf(), "A new node should be a leaf"
    assert root_node.parent is None, "Root node's parent should be None"
    assert root_node.color == player_color, "Node color should be set correctly"
    assert root_node.game is game_instance, "Node should hold the correct game instance"
    assert root_node.level == 0, "Root node's level should be 0"
    assert not root_node.prunning, "Pruning should be False by default or as set"
    assert (
        root_node.children == []
    ), "Initial children should be an empty list"  # children is initialized as [] then turned into defaultdict in expand


def test_child_node_initial_properties():
    """
    Tests the initial properties of a child StateNode.
    """
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    game_instance = game_instance = Game(players)
    parent_color = Color.RED
    child_color = (
        Color.RED
    )  # Typically the MCTS player\'s color remains the same for nodes it creates

    parent_node = StateNode(color=parent_color, game=game_instance, parent=None)
    parent_node.level = 5  # Manually set for testing child\'s level calculation

    # Create a new game state for the child, perhaps by copying or a new instance
    child_game_instance = Game(players)

    child_node = StateNode(
        color=child_color, game=child_game_instance, parent=parent_node, prunning=True
    )

    assert child_node.wins == 0, "Initial wins for child should be 0"
    assert child_node.visits == 0, "Initial visits for child should be 0"
    assert child_node.is_leaf(), "New child node should be a leaf"
    assert (
        child_node.parent is parent_node
    ), "Child node's parent should be set correctly"
    assert child_node.color == child_color, "Child node's color should be set correctly"
    assert (
        child_node.game is child_game_instance
    ), "Child node should hold its own game instance"
    assert (
        child_node.level == parent_node.level + 1
    ), "Child node's level should be parent's level + 1"
    assert child_node.prunning, "Child node's pruning status should be set correctly"
    assert (
        child_node.children == []
    ), "Initial children for child node should be an empty list"


def _terminal_1v1_game(seed=7):
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    game.play()
    assert game.winning_color() is not None
    return game


def _midgame_1v1_game(seed=3, ticks=20):
    random.seed(seed)
    game = Game([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)])
    for _ in range(ticks):
        if game.winning_color() is not None:
            break
        game.play_tick()
    assert game.winning_color() is None
    return game


def test_terminal_value_is_read_from_selected_leaf():
    """A terminal win at the selected leaf must be credited to the searcher.

    Regression test: the old code read ``self.game.winning_color()`` from the
    (non-terminal) root, so terminal wins were scored as losses.
    """
    terminal_game = _terminal_1v1_game()
    winner = terminal_game.winning_color()

    # Fresh, non-terminal root: if the winner were read from the root, it would
    # be ``None`` and the win would not be credited.
    root_game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    root = StateNode(winner, root_game, None)

    action = root_game.playable_actions[0]
    leaf = StateNode(winner, terminal_game, root)
    root.children = {action: [(leaf, 1.0)]}

    root.run_simulation()

    assert leaf.wins == 1.0, "winning terminal leaf must be credited 1.0"
    assert root.wins == 1.0, "the win must backpropagate to the root"


def test_terminal_loss_is_scored_zero():
    terminal_game = _terminal_1v1_game()
    winner = terminal_game.winning_color()
    loser = Color.RED if winner == Color.BLUE else Color.BLUE

    root_game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    root = StateNode(loser, root_game, None)
    action = root_game.playable_actions[0]
    leaf = StateNode(loser, terminal_game, root)
    root.children = {action: [(leaf, 1.0)]}

    root.run_simulation()

    assert leaf.wins == 0.0
    assert root.wins == 0.0


def test_decide_returns_legal_action():
    game = _midgame_1v1_game()
    player = MCTSPlayer(game.state.current_color(), num_simulations=10)
    action = player.decide(game, game.playable_actions)
    assert action in game.playable_actions


def test_decide_with_pruning_returns_legal_action():
    """Pruning must not select a pruned-away action (old bug crashed here)."""
    random.seed(1)
    # A fresh game is in the initial build phase, where pruning is active.
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    player = MCTSPlayer(game.state.current_color(), num_simulations=15, prunning=True)
    action = player.decide(game, game.playable_actions)
    assert action in game.playable_actions


def _make_node(node_color, game, children_specs):
    node = StateNode(node_color, game, None)
    node.children = {}
    total_visits = 0
    for action, (wins, visits) in children_specs.items():
        child = StateNode(node_color, game, node)
        child.wins = wins
        child.visits = visits
        node.children[action] = [(child, 1.0)]
        total_visits += visits
    node.visits = total_visits
    return node


def test_selection_is_opponent_aware():
    """On our turn we maximize our value; on the opponent's turn they minimize it."""
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    to_move = game.state.current_color()
    other = Color.BLUE if to_move == Color.RED else Color.RED

    a_good, a_bad = game.playable_actions[0], game.playable_actions[1]
    specs = {a_good: (9, 10), a_bad: (1, 10)}  # a_good is better for the searcher

    our_node = _make_node(to_move, game, specs)
    assert our_node._select_action(exploration=False) == a_good

    opp_node = _make_node(other, game, specs)
    assert opp_node._select_action(exploration=False) == a_bad


def test_best_action_by_visits_picks_most_visited():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    a0, a1 = game.playable_actions[0], game.playable_actions[1]
    node = _make_node(game.state.current_color(), game, {a0: (1, 3), a1: (1, 20)})
    assert node.best_action_by_visits() == a1
