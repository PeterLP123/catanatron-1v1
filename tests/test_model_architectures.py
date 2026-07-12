from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catanatron.gym.model_architectures import (  # noqa: E402
    ActionConditionedScorer,
    BoardTensorEncoder,
)


def test_action_conditioned_scorer_supports_full_and_legal_only_scores():
    model = ActionConditionedScorer(12, 20, hidden_sizes=(16,), embedding_dim=8)
    observations = torch.randn(3, 12)
    legal = torch.tensor([[1, 4, 9], [2, 3, 5], [0, 8, 19]])

    full = model(observations)
    candidates = model(observations, legal)

    assert full.shape == (3, 20)
    assert candidates.shape == (3, 3)
    torch.testing.assert_close(candidates, torch.gather(full, 1, legal))


def test_board_tensor_encoder_fuses_board_and_numeric_state():
    encoder = BoardTensorEncoder(7, 15, output_dim=32)
    encoded = encoder(torch.randn(4, 7, 21, 11), torch.randn(4, 15))

    assert encoded.shape == (4, 32)
