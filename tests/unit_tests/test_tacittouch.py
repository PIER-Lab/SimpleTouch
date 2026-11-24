# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma, model
from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from scripts.tacittouch.objective import (
    _predict_at_query,
    distillation_losses,
    vision_observation,
)
from scripts.tacittouch.settings import TacitTouchSettings
from scripts.tacittouch.training import TacitTouchActionModel
from scripts.univtac.reference_runtime import reference_file, reference_root


class ActionBlock(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(float(shift)))

    def forward(self, streams):
        return [None, streams[1] + self.shift], ()


class TinyPolicy(nn.Module):
    """Expose the same suffix interface as Pi0 with a closed-form velocity."""

    action_dim = 1
    action_horizon = 2
    embed_dtype = torch.float32

    def __init__(self, *, tactile, shift):
        super().__init__()
        self.tactile_expert_enabled = tactile
        self.llm = nn.Module()
        self.llm.layers = nn.ModuleList([ActionBlock(shift), ActionBlock(shift)])
        self.calls = []
        self.demo_calls = 0

    def build_prefix_cache(self, observation):
        assert (observation.tactile_images is not None) == self.tactile_expert_enabled
        return None, torch.ones((1, 1), dtype=torch.bool), ()

    def encode_policy_tactile(self, observation):
        return observation.tactile_images["left"]

    def build_tactile_expert_cache(self, observation, tactile, mask, cache):
        return mask, cache

    def run_suffix(self, observation, state, time, cache, mask):
        self.calls.append(
            (state, time, observation.images["head"], torch.is_grad_enabled())
        )
        # A non-action token verifies that only the final H tokens match.
        tokens = torch.cat([state.new_full((1, 1, 1), 100.0), state], dim=1)
        for block in self.llm.layers:
            (_, tokens), _ = block([None, tokens])
        return 2 * tokens[:, -self.action_horizon :]

    def velocity_from_suffix(self, suffix):
        return suffix

    def compute_loss(self, observation, actions, *, train):
        assert train and observation.tactile_images is None
        self.demo_calls += 1
        return actions.square().mean(dim=-1) + self.llm.layers[0].shift


@pytest.fixture
def observation(monkeypatch):
    def preprocess(obs, *, train):
        assert train
        obs.images = {"head": obs.images["head"] + 1}
        return obs

    monkeypatch.setattr(model, "preprocess_observation", preprocess)
    return model.Observation(
        images={"head": torch.zeros((1, 3, 2, 2))},
        image_masks={"head": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros((1, 1)),
        tactile_images={"left": torch.ones((1, 3, 2, 2))},
        tactile_masks={"left": torch.ones(1, dtype=torch.bool)},
    )


def test_shared_query_stops_prefix_and_teacher_gradients(observation, monkeypatch):
    student = TinyPolicy(tactile=False, shift=1)
    teacher = TinyPolicy(tactile=True, shift=3).eval().requires_grad_(False)
    noise = torch.full((1, 2, 1), 5.0, requires_grad=True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast(index, src):
        assert src == 0
        index.fill_(2)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    velocity, features = distillation_losses(
        student,
        teacher,
        observation,
        num_steps=10,
        collect_features=True,
        noise=noise,
        query_index=0,
    )

    assert len(student.calls) == 3 and len(teacher.calls) == 1
    assert [call[3] for call in student.calls] == [False, False, True]
    state, time, image, grad = teacher.calls[0]
    assert not grad and not state.requires_grad
    assert state is student.calls[-1][0] and time is student.calls[-1][1]
    assert image is student.calls[-1][2]
    torch.testing.assert_close(state, torch.full_like(noise, 2.48))
    torch.testing.assert_close(time, torch.tensor([0.8]))
    # Velocity gets the final factor of 2. Raw features are captured before it.
    torch.testing.assert_close(velocity, torch.tensor(64.0))
    torch.testing.assert_close(features, torch.tensor((4.0 + 16.0) / 2))
    (velocity + 0.01 * features).backward()
    assert noise.grad is None
    assert all(param.grad is None for param in teacher.parameters())
    assert all(param.grad is not None for param in student.parameters())
    assert not any(layer._forward_hooks for layer in student.llm.layers)


def test_demo_pass_and_student_checkpoint(observation):
    student = TinyPolicy(tactile=False, shift=1)
    teacher = TinyPolicy(tactile=True, shift=3)
    settings = TacitTouchSettings(
        teacher_checkpoint="teacher", tactile_encoder_path="t3"
    )
    policy = TacitTouchActionModel(
        student, teacher, settings, num_steps=10, action_env_dim=1
    )
    same_student = OpenPiPytorchSFTActionModel(student, num_steps=10, action_env_dim=1)
    policy.train()
    assert not teacher.training
    assert set(policy.state_dict()) == set(same_student.state_dict())
    assert set(policy.parameters()) == set(student.parameters())

    loss = policy.sft_forward((observation, torch.full((1, 2, 1), 2.0)))
    assert student.demo_calls == 1 and teacher.demo_calls == 0
    torch.testing.assert_close(loss["demonstration_loss"], torch.tensor(5.0))
    torch.testing.assert_close(
        loss["loss"],
        loss["action_distillation_loss"]
        + 0.01 * loss["feature_distillation_loss"]
        + loss["demonstration_loss"],
    )
    loss["loss"].backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_public_recipe_keeps_standard_simpletouch_opt_in(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", (repo / "examples/sft").as_posix())
    with initialize_config_dir(
        config_dir=str(repo / "examples/sft/config"), version_base="1.1"
    ):
        student = compose(
            config_name="simpletac_univtac", overrides=["+tacittouch=univtac"]
        )
        standard = compose(config_name="simpletac_univtac")
    assert student.actor.model.tacittouch.demonstration_loss_weight == 1
    assert student.actor.model.tacittouch.feature_loss_weight == 0.01
    assert student.actor.model.num_steps == 10
    assert student.actor.model.num_action_chunks == 50
    assert not student.actor.model.openpi.tactile_expert_enabled
    assert not student.data.tactile_expert_future_cache_path
    assert student.actor.global_batch_size == 64
    assert student.actor.micro_batch_size == 8
    assert student.runner.max_steps == 20000
    assert "tacittouch" not in standard.actor.model
    assert standard.actor.model.openpi.tactile_expert_enabled


def test_captures_all_eighteen_real_gemma_blocks(observation):
    class GemmaPolicy(TinyPolicy):
        def __init__(self):
            super().__init__(tactile=False, shift=0)
            config = gemma.Config(
                width=8,
                depth=18,
                mlp_dim=16,
                num_heads=2,
                num_kv_heads=1,
                head_dim=4,
            )
            self.llm = gemma.Module(
                [config, config],
                embed_dtype="float32",
                use_gradient_checkpointing=False,
            )

        def run_suffix(self, observation, state, time, cache, mask):
            tokens = state.expand(-1, -1, 8)
            outputs, _ = self.llm(
                [None, tokens],
                positions=torch.arange(2).unsqueeze(0),
                mask=torch.ones((1, 2, 2), dtype=torch.bool),
            )
            return outputs[1]

    policy = GemmaPolicy()
    prediction, features = _predict_at_query(
        policy,
        vision_observation(observation),
        torch.tensor([[[1.0], [2.0]]]),
        torch.ones(1),
        collect_features=True,
    )
    assert len(features) == 18
    assert all(feature.shape == (1, 2, 8) for feature in features)
    normalized, _ = policy.llm.final_norms[1](features[-1], None)
    torch.testing.assert_close(prediction, normalized)
    assert not any(block._forward_hooks for block in policy.llm.layers)


def test_tacittouch_insert_tube_uses_absolute_lateral_displacement():
    root = reference_root()
    name = "envs/insert_tube.py"
    reference = reference_file(root, name)
    patched = reference_file(root, name, "tacittouch")
    assert reference == (root / name).read_bytes()
    assert (
        reference_file(root, "envs/lift_can.py", "tacittouch")
        == (root / "envs/lift_can.py").read_bytes()
    )
    assert patched != reference
    namespace = {}
    # Load only the success method, so the fixture needs no simulator.
    import ast

    tree = ast.parse(patched)
    check = next(
        child
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == "check_success"
    )
    exec(
        compile(ast.Module(body=[check], type_ignores=[]), name, "exec"),
        {"np": np},
        namespace,
    )

    class Pose:
        def __init__(self):
            self.p = np.array([0.004, -0.004, -0.05])

        def rebase(self, frame):
            return self

        def tolist(self):
            return self.p.tolist()

        def __getitem__(self, index):
            return self.p[index]

        def to_transformation_matrix(self):
            return np.eye(4)

    pose = Pose()
    from types import SimpleNamespace

    task = SimpleNamespace(
        prism=SimpleNamespace(get_pose=lambda: pose),
        hole_pose=object(),
        _robot_manager=SimpleNamespace(get_gripper_center_pose=lambda: object()),
        metadata={},
        origin_inhand_pose=pose.p.copy(),
    )
    success = namespace["check_success"]
    assert success(task)
    pose.p[0] = -0.020
    assert not success(task)
    pose.p[0], pose.p[1] = 0.004, 0.005
    assert not success(task)
