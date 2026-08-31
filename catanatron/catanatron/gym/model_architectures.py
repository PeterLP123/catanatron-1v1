"""Structured policy components for BC and search distillation experiments.

These modules are deliberately independent of SB3. The stable PPO path keeps
its vector MLP, while BC and expert-iteration experiments can share parameters
across action identities and train state-value heads from trajectory outcomes.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Sequence

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
        self.policy_head = ActionConditionedHead(embedding_dim, n_actions)

    @property
    def action_embedding(self) -> nn.Embedding:
        return self.policy_head.action_embedding

    @property
    def action_bias(self) -> nn.Embedding:
        return self.policy_head.action_bias

    def forward(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.policy_head(self.state_encoder(observations), action_ids)


class ActionConditionedHead(nn.Module):
    """Dot-product policy head over all actions or a legal candidate subset."""

    def __init__(self, embedding_dim: int, n_actions: int) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.action_embedding = nn.Embedding(n_actions, embedding_dim)
        self.action_bias = nn.Embedding(n_actions, 1)

    def forward(
        self, state: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if action_ids is None:
            scores = state @ self.action_embedding.weight.transpose(0, 1)
            return scores + self.action_bias.weight.squeeze(-1)
        embeddings = self.action_embedding(action_ids)
        bias = self.action_bias(action_ids).squeeze(-1)
        return (embeddings * state.unsqueeze(1)).sum(dim=-1) + bias


class PolicyValueOutput(NamedTuple):
    policy_logits: torch.Tensor
    win_value: torch.Tensor
    vp_margin: torch.Tensor


class OutcomeCriticOutput(NamedTuple):
    win_logit: torch.Tensor
    vp_margin: torch.Tensor


def factored_feature_groups(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Partition the stable vector schema into semantic Catan state groups."""
    groups: dict[str, list[int]] = {
        "edges": [],
        "nodes": [],
        "tiles": [],
        "ports": [],
        "global": [],
    }
    for index, raw_name in enumerate(feature_names):
        name = raw_name.removeprefix("F_")
        if name.startswith("EDGE("):
            group = "edges"
        elif name.startswith("NODE"):
            group = "nodes"
        elif name.startswith("TILE"):
            group = "tiles"
        elif name.startswith("PORT"):
            group = "ports"
        else:
            group = "global"
        groups[group].append(index)
    empty = [name for name, indices in groups.items() if not indices]
    if empty:
        raise ValueError(f"Factored feature schema has empty groups: {empty}")
    return {name: tuple(indices) for name, indices in groups.items()}


class FactoredPolicyValueNet(nn.Module):
    """Policy/value network that preserves the main Catan state factorization.

    The current 614-feature datasets do not all contain the optional board
    tensor. This network therefore groups the stable vector schema into edges,
    nodes, tiles, ports and global state, encodes each separately, and fuses the
    result before scoring learned action identities. It can train on every
    existing BC/DAgger shard without flattening all semantics into one layer.
    """

    group_names = ("edges", "nodes", "tiles", "ports", "global")

    def __init__(
        self,
        feature_names: Sequence[str],
        n_actions: int,
        *,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        groups = factored_feature_groups(feature_names)
        self.obs_dim = len(feature_names)
        self.n_actions = int(n_actions)
        self.embedding_dim = int(embedding_dim)
        branch_dim = max(32, embedding_dim // 2)
        branch_hidden = max(64, embedding_dim)
        self.encoders = nn.ModuleDict()
        for name in self.group_names:
            indices = torch.as_tensor(groups[name], dtype=torch.long)
            self.register_buffer(f"_{name}_indices", indices)
            self.encoders[name] = _mlp(len(groups[name]), (branch_hidden,), branch_dim)
        self.fusion = _mlp(
            branch_dim * len(self.group_names),
            (max(256, embedding_dim * 2),),
            embedding_dim,
        )
        self.policy_head = ActionConditionedHead(embedding_dim, n_actions)
        self.win_head = nn.Linear(embedding_dim, 1)
        self.vp_margin_head = nn.Linear(embedding_dim, 1)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected observations with shape [batch, {self.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )
        parts = [
            self.encoders[name](
                observations.index_select(1, getattr(self, f"_{name}_indices"))
            )
            for name in self.group_names
        ]
        return self.fusion(torch.cat(parts, dim=-1))

    def policy_value(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> PolicyValueOutput:
        state = self.encode(observations)
        return PolicyValueOutput(
            policy_logits=self.policy_head(state, action_ids),
            win_value=torch.tanh(self.win_head(state)).squeeze(-1),
            vp_margin=self.vp_margin_head(state).squeeze(-1),
        )

    def forward(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.policy_value(observations, action_ids).policy_logits


class FactoredOutcomeCritic(nn.Module):
    """Value-only critic over the stable semantic feature groups.

    This deliberately has no policy head. It can be trained from terminal
    outcomes without moving the retained policy parameters, then admitted to a
    reranking/search experiment only after beating public-score baselines on
    whole-game validation and test splits.
    """

    group_names = ("edges", "nodes", "tiles", "ports", "global")

    def __init__(
        self, feature_names: Sequence[str], *, embedding_dim: int = 128
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        groups = factored_feature_groups(feature_names)
        self.obs_dim = len(feature_names)
        self.embedding_dim = int(embedding_dim)
        branch_dim = max(32, embedding_dim // 2)
        branch_hidden = max(64, embedding_dim)
        self.encoders = nn.ModuleDict()
        for name in self.group_names:
            indices = torch.as_tensor(groups[name], dtype=torch.long)
            self.register_buffer(f"_{name}_indices", indices)
            self.encoders[name] = _mlp(len(groups[name]), (branch_hidden,), branch_dim)
        self.fusion = _mlp(
            branch_dim * len(self.group_names),
            (max(256, embedding_dim * 2),),
            embedding_dim,
        )
        self.win_head = nn.Linear(embedding_dim, 1)
        self.vp_margin_head = nn.Linear(embedding_dim, 1)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected observations with shape [batch, {self.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )
        parts = [
            self.encoders[name](
                observations.index_select(1, getattr(self, f"_{name}_indices"))
            )
            for name in self.group_names
        ]
        return self.fusion(torch.cat(parts, dim=-1))

    def forward(self, observations: torch.Tensor) -> OutcomeCriticOutput:
        state = self.encode(observations)
        return OutcomeCriticOutput(
            win_logit=self.win_head(state).squeeze(-1),
            vp_margin=self.vp_margin_head(state).squeeze(-1),
        )


def _indexed_entity_features(
    feature_names: Sequence[str], pattern: str
) -> tuple[dict[object, tuple[int, ...]], tuple[str, ...]]:
    """Parse repeated entity features and require one stable suffix schema."""
    compiled = re.compile(pattern)
    records: dict[object, dict[str, int]] = {}
    for feature_index, raw_name in enumerate(feature_names):
        match = compiled.fullmatch(raw_name.removeprefix("F_"))
        if match is None:
            continue
        raw_entity, suffix = match.groups()
        entity: object
        if "," in raw_entity:
            left, right = raw_entity.split(",")
            entity = tuple(sorted((int(left), int(right))))
        else:
            entity = int(raw_entity)
        if suffix in records.setdefault(entity, {}):
            raise ValueError(f"Duplicate feature for entity {entity}: {suffix}")
        records[entity][suffix] = feature_index
    if not records:
        raise ValueError(f"Feature schema has no entities matching {pattern!r}")
    suffixes = tuple(sorted(next(iter(records.values()))))
    for entity, values in records.items():
        if tuple(sorted(values)) != suffixes:
            raise ValueError(
                f"Entity {entity} has inconsistent feature suffixes: "
                f"{tuple(sorted(values))} != {suffixes}"
            )
    return {
        entity: tuple(values[suffix] for suffix in suffixes)
        for entity, values in records.items()
    }, suffixes


class SpatialEdgeResidualPolicy(nn.Module):
    """An MLP policy plus a shared topology-aware residual for road actions.

    The base MLP is byte-compatible with existing checkpoints. The residual
    binds each BUILD_ROAD action to its edge ownership, endpoint buildings,
    adjacent tile state, and a learned positional embedding. Its output layer
    starts at exactly zero, so loading and freezing a retained MLP produces an
    epoch-0 policy with identical logits before the road treatment is trained.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        n_actions: int,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        from catanatron.gym.envs.action_space import get_action_array
        from catanatron.models.board import base_map
        from catanatron.models.enums import ActionType
        from catanatron.models.player import Color

        action_codec = get_action_array((Color.BLUE, Color.RED), "BASE")
        if n_actions != len(action_codec):
            raise ValueError(
                f"Spatial edge policy expects {len(action_codec)} BASE actions, "
                f"got {n_actions}"
            )
        road_actions = [
            (action_index, tuple(sorted(value)))
            for action_index, (action_type, value) in enumerate(action_codec)
            if action_type == ActionType.BUILD_ROAD
        ]
        edge_features, _ = _indexed_entity_features(
            feature_names, r"EDGE\((\d+,\s*\d+)\)_(.+)"
        )
        node_features, _ = _indexed_entity_features(feature_names, r"NODE(\d+)_(.+)")
        tile_features, _ = _indexed_entity_features(feature_names, r"TILE(\d+)_(.+)")
        road_edges = [edge for _, edge in road_actions]
        if set(road_edges) != set(edge_features):
            raise ValueError("Road actions and edge feature identities do not match")

        edge_feature_indices = []
        endpoint_node_feature_indices = []
        tile_ids = sorted(tile_features)
        tile_position = {tile_id: index for index, tile_id in enumerate(tile_ids)}
        edge_tile_incidence = torch.zeros(len(road_edges), len(tile_ids))
        for edge_position, edge in enumerate(road_edges):
            edge_feature_indices.append(edge_features[edge])
            endpoint_node_feature_indices.append(
                (*node_features[edge[0]], *node_features[edge[1]])
            )
            adjacent_tile_ids = {
                tile.id for node_id in edge for tile in base_map.adjacent_tiles[node_id]
            }
            for tile_id in adjacent_tile_ids:
                edge_tile_incidence[edge_position, tile_position[tile_id]] = 1.0
            edge_tile_incidence[edge_position] /= len(adjacent_tile_ids)

        self.obs_dim = len(feature_names)
        self.n_actions = int(n_actions)
        self.embedding_dim = int(embedding_dim)
        self.base_policy = _mlp(self.obs_dim, hidden_sizes, n_actions)
        edge_width = len(edge_feature_indices[0])
        endpoint_width = len(endpoint_node_feature_indices[0])
        tile_width = len(next(iter(tile_features.values())))
        local_width = edge_width + endpoint_width + tile_width
        context_width = max(128, embedding_dim * 2)
        local_hidden = max(64, embedding_dim)
        self.context_encoder = _mlp(self.obs_dim, (context_width,), embedding_dim)
        self.local_encoder = _mlp(local_width, (local_hidden,), embedding_dim)
        self.edge_position = nn.Embedding(len(road_edges), embedding_dim)
        self.delta_head = nn.Linear(embedding_dim * 3, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.register_buffer(
            "road_action_indices",
            torch.as_tensor([index for index, _ in road_actions], dtype=torch.long),
        )
        self.register_buffer(
            "edge_feature_indices",
            torch.as_tensor(edge_feature_indices, dtype=torch.long),
        )
        self.register_buffer(
            "endpoint_node_feature_indices",
            torch.as_tensor(endpoint_node_feature_indices, dtype=torch.long),
        )
        self.register_buffer(
            "tile_feature_indices",
            torch.as_tensor(
                [tile_features[tile_id] for tile_id in tile_ids], dtype=torch.long
            ),
        )
        self.register_buffer("edge_tile_incidence", edge_tile_incidence)

    def load_base_policy_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base_policy.load_state_dict(state_dict)

    def freeze_base_policy(self) -> None:
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

    def road_deltas(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        edge_state = observations[:, self.edge_feature_indices]
        endpoint_state = observations[:, self.endpoint_node_feature_indices]
        tile_state = observations[:, self.tile_feature_indices]
        adjacent_tile_state = torch.einsum(
            "et,btf->bef", self.edge_tile_incidence, tile_state
        )
        local_state = torch.cat(
            (edge_state, endpoint_state, adjacent_tile_state), dim=-1
        )
        local_embedding = self.local_encoder(
            local_state.reshape(-1, local_state.shape[-1])
        ).reshape(batch_size, len(self.road_action_indices), self.embedding_dim)
        local_embedding = local_embedding + self.edge_position.weight.unsqueeze(0)
        context = (
            self.context_encoder(observations).unsqueeze(1).expand_as(local_embedding)
        )
        interaction = torch.cat(
            (context, local_embedding, context * local_embedding), dim=-1
        )
        return self.delta_head(interaction).squeeze(-1)

    def forward(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected observations with shape [batch, {self.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )
        logits = self.base_policy(observations).index_add(
            1, self.road_action_indices, self.road_deltas(observations)
        )
        if action_ids is None:
            return logits
        return torch.gather(logits, 1, action_ids)


class SpatialRobberResidualPolicy(nn.Module):
    """An MLP policy plus a shared tile-aware residual for robber actions.

    The base MLP is byte-compatible with existing checkpoints. Each residual
    candidate binds a MOVE_ROBBER action to its destination tile, whether it
    selects a victim, shared global context, and a learned tile position. Raw
    BLUE/RED victim identity is intentionally collapsed because vector features
    are expressed from the acting player's P0/P1 perspective.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        n_actions: int,
        *,
        hidden_sizes: Sequence[int] = (512, 512),
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        from catanatron.gym.envs.action_space import get_action_array
        from catanatron.models.board import base_map
        from catanatron.models.enums import ActionType
        from catanatron.models.player import Color

        action_codec = get_action_array((Color.BLUE, Color.RED), "BASE")
        if n_actions != len(action_codec):
            raise ValueError(
                f"Spatial robber policy expects {len(action_codec)} BASE actions, "
                f"got {n_actions}"
            )
        robber_actions = [
            (action_index, coordinates, victim)
            for action_index, (action_type, value) in enumerate(action_codec)
            if action_type == ActionType.MOVE_ROBBER
            for coordinates, victim in (value,)
        ]
        tile_features, _ = _indexed_entity_features(feature_names, r"TILE(\d+)_(.+)")
        tile_ids = sorted(tile_features)
        if set(tile_ids) != {tile.id for tile in base_map.land_tiles.values()}:
            raise ValueError(
                "Robber destinations and tile feature identities do not match"
            )
        tile_position = {tile_id: index for index, tile_id in enumerate(tile_ids)}
        action_tile_ids = [
            base_map.land_tiles[coordinates].id for _, coordinates, _ in robber_actions
        ]

        self.obs_dim = len(feature_names)
        self.n_actions = int(n_actions)
        self.embedding_dim = int(embedding_dim)
        self.base_policy = _mlp(self.obs_dim, hidden_sizes, n_actions)
        tile_width = len(next(iter(tile_features.values())))
        context_width = max(128, embedding_dim * 2)
        local_hidden = max(64, embedding_dim)
        self.context_encoder = _mlp(self.obs_dim, (context_width,), embedding_dim)
        self.local_encoder = _mlp(tile_width + 1, (local_hidden,), embedding_dim)
        self.tile_position = nn.Embedding(len(tile_ids), embedding_dim)
        self.delta_head = nn.Linear(embedding_dim * 3, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.register_buffer(
            "robber_action_indices",
            torch.as_tensor(
                [index for index, _, _ in robber_actions], dtype=torch.long
            ),
        )
        self.register_buffer(
            "robber_tile_feature_indices",
            torch.as_tensor(
                [tile_features[tile_id] for tile_id in action_tile_ids],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "robber_tile_positions",
            torch.as_tensor(
                [tile_position[tile_id] for tile_id in action_tile_ids],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "robber_victim_present",
            torch.as_tensor(
                [[float(victim is not None)] for _, _, victim in robber_actions]
            ),
        )

    def load_base_policy_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base_policy.load_state_dict(state_dict)

    def freeze_base_policy(self) -> None:
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

    def robber_deltas(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        tile_state = observations[:, self.robber_tile_feature_indices]
        victim_present = self.robber_victim_present.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        local_state = torch.cat((tile_state, victim_present), dim=-1)
        local_embedding = self.local_encoder(
            local_state.reshape(-1, local_state.shape[-1])
        ).reshape(batch_size, len(self.robber_action_indices), self.embedding_dim)
        local_embedding = local_embedding + self.tile_position(
            self.robber_tile_positions
        ).unsqueeze(0)
        context = (
            self.context_encoder(observations).unsqueeze(1).expand_as(local_embedding)
        )
        interaction = torch.cat(
            (context, local_embedding, context * local_embedding), dim=-1
        )
        return self.delta_head(interaction).squeeze(-1)

    def forward(
        self, observations: torch.Tensor, action_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected observations with shape [batch, {self.obs_dim}], "
                f"got {tuple(observations.shape)}"
            )
        logits = self.base_policy(observations).index_add(
            1, self.robber_action_indices, self.robber_deltas(observations)
        )
        if action_ids is None:
            return logits
        return torch.gather(logits, 1, action_ids)


def build_bc_policy(
    architecture: str,
    feature_names: Sequence[str],
    n_actions: int,
    *,
    hidden_sizes: Sequence[int] = (512, 512),
    embedding_dim: int = 128,
) -> nn.Module:
    """Build a backward-compatible flat or structured Torch BC policy."""
    if architecture == "mlp":
        return _mlp(len(feature_names), hidden_sizes, n_actions)
    if architecture == "action_conditioned":
        return ActionConditionedScorer(
            len(feature_names),
            n_actions,
            hidden_sizes=hidden_sizes,
            embedding_dim=embedding_dim,
        )
    if architecture == "factored_policy_value":
        return FactoredPolicyValueNet(
            feature_names, n_actions, embedding_dim=embedding_dim
        )
    if architecture == "spatial_edge_residual":
        return SpatialEdgeResidualPolicy(
            feature_names,
            n_actions,
            hidden_sizes=hidden_sizes,
            embedding_dim=embedding_dim,
        )
    if architecture == "spatial_robber_residual":
        return SpatialRobberResidualPolicy(
            feature_names,
            n_actions,
            hidden_sizes=hidden_sizes,
            embedding_dim=embedding_dim,
        )
    raise ValueError(f"Unknown BC architecture {architecture!r}")


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
