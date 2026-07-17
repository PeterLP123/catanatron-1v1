from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from catanatron.gym.anchored_ppo import (
    AnchoredMaskablePPO,
    legal_action_forward_kl,
)


def test_legal_action_forward_kl_ignores_illegal_logits_and_gradients():
    reference = torch.tensor([[2.0, -4.0, 0.0]], dtype=torch.float64)
    current = torch.tensor([[0.0, 100.0, 2.0]], dtype=torch.float64, requires_grad=True)
    masks = torch.tensor([[True, False, True]])

    loss = legal_action_forward_kl(reference, current, masks)
    loss.backward()

    changed_illegal = current.detach().clone()
    changed_illegal[0, 1] = -100.0
    assert loss.item() > 0
    assert legal_action_forward_kl(
        reference, changed_illegal, masks
    ).item() == pytest.approx(loss.item())
    assert current.grad is not None
    assert current.grad[0, 1].item() == 0.0


def test_legal_action_forward_kl_is_zero_for_matching_policy():
    logits = torch.tensor([[1.0, 2.0, -3.0], [0.0, 4.0, 2.0]])
    masks = torch.tensor([[True, True, False], [False, True, True]])

    assert legal_action_forward_kl(logits, logits, masks).item() == pytest.approx(0.0)


def test_anchor_trains_and_saves_an_ordinary_inference_checkpoint(tmp_path):
    env = ActionMasker(
        gym.make("CartPole-v1"),
        lambda wrapped: np.ones(wrapped.action_space.n, dtype=bool),
    )
    model = AnchoredMaskablePPO(
        "MlpPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        bc_anchor_coef=0.1,
    )
    model.set_anchor_from_current_policy()
    assert model.anchor_policy is not None
    before = [
        parameter.detach().clone() for parameter in model.anchor_policy.parameters()
    ]

    with torch.no_grad():
        next(model.policy.parameters()).add_(1.0)

    assert all(
        not parameter.requires_grad for parameter in model.anchor_policy.parameters()
    )
    assert all(
        torch.equal(saved, current)
        for saved, current in zip(before, model.anchor_policy.parameters())
    )
    assert "anchor_policy" in model._excluded_save_params()

    model.learn(total_timesteps=4)
    checkpoint = tmp_path / "anchored.zip"
    model.save(checkpoint)
    loaded = MaskablePPO.load(checkpoint)
    assert not hasattr(loaded, "anchor_policy")


def test_anchor_coefficient_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        AnchoredMaskablePPO("MlpPolicy", gym.make("CartPole-v1"), bc_anchor_coef=-0.1)
