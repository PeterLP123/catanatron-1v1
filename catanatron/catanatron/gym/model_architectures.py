"""Structured policy components for search distillation experiments.

These modules are deliberately independent of SB3.  The stable PPO path keeps
its vector MLP, while expert-iteration experiments can share parameters across
action identities and encode the existing board tensor without another custom
policy rewrite.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(current, width), nn.ReLU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class ActionConditionedScorer(nn.Module):
    """Score action IDs against a shared state embedding.

    Unlike a flat 332-way final layer, this head learns one state encoder and
    one reusable embedding per action.  ``action_ids`` may contain only legal
    candidates, which makes the same module suitable for listwise distillation.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_sizes: Sequence[int] = (512, 256),
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.state_encoder = _mlp(obs_dim, hidden_sizes, embedding_dim)
        self.action_embedding = nn.Embedding(n_actions, embedding_dim)
        self.action_bias = nn.Embedding(n_actions, 1)

    def forward(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        state = self.state_encoder(observations)
        if action_ids is None:
            scores = state @ self.action_embedding.weight.transpose(0, 1)
            return scores + self.action_bias.weight.squeeze(-1)
        embeddings = self.action_embedding(action_ids)
        bias = self.action_bias(action_ids).squeeze(-1)
        return (embeddings * state.unsqueeze(1)).sum(dim=-1) + bias


class BoardTensorEncoder(nn.Module):
    """Encode the existing board tensor plus numeric public state."""

    def __init__(
        self,
        board_channels: int,
        numeric_dim: int,
        *,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )
        self.fusion = _mlp(32 * 3 * 3 + numeric_dim, (512,), output_dim)

    def forward(self, board: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        board_state = self.board(board)
        return self.fusion(torch.cat((board_state, numeric), dim=-1))
