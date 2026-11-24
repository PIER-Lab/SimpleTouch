# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""TacitTouch's shared-query action and feature distillation."""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_pytorch.pi0_model import model
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


def vision_observation(observation: model.Observation) -> model.Observation:
    return dataclasses.replace(
        observation,
        tactile_images=None,
        tactile_masks=None,
        tactile_future_images=None,
        tactile_future_masks=None,
        tactile_future_target_valid=None,
        tactile_world_latents=None,
        tactile_world_future_latents=None,
    )


def _action_context(policy: Pi0, observation: model.Observation):
    _, mask, cache = policy.build_prefix_cache(observation)
    if policy.tactile_expert_enabled:
        tactile = policy.encode_policy_tactile(observation)
        if tactile is None:
            raise ValueError(
                "TacitTouch needs measured fingertip images for the teacher"
            )
        mask, cache = policy.build_tactile_expert_cache(
            observation, tactile, mask, cache
        )
    return cache, mask


def _predict_at_query(
    policy: Pi0,
    observation: model.Observation,
    state: torch.Tensor,
    time: torch.Tensor,
    *,
    collect_features: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    cache, mask = _action_context(policy, observation)
    features: list[torch.Tensor] = []

    def capture_action_tokens(_layer, _inputs, output) -> None:
        streams, _ = output
        features.append(streams[1][:, -policy.action_horizon :])

    # Block outputs precede the final Gemma normalization. The scoped hooks
    # collect only the action expert's tokens for this suffix pass.
    with ExitStack() as stack:
        if collect_features:
            for block in policy.llm.layers:
                stack.enter_context(block.register_forward_hook(capture_action_tokens))
        suffix = policy.run_suffix(observation, state, time, cache, mask)
    return policy.velocity_from_suffix(suffix), tuple(features)


def distillation_losses(
    student: Pi0,
    teacher: Pi0,
    observation: model.Observation,
    *,
    num_steps: int,
    collect_features: bool,
    noise: torch.Tensor | None = None,
    query_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the two policies at one detached student-trajectory query."""
    if (
        student is teacher
        or student.tactile_expert_enabled
        or not teacher.tactile_expert_enabled
    ):
        raise ValueError(
            "TacitTouch requires a visual student and a separate tactile teacher"
        )
    if (student.action_horizon, student.action_dim) != (
        teacher.action_horizon,
        teacher.action_dim,
    ):
        raise ValueError("Teacher and student action shapes differ")
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    visual = vision_observation(observation)
    teacher_observation = dataclasses.replace(
        visual,
        tactile_images=observation.tactile_images,
        tactile_masks=observation.tactile_masks,
    )
    # Sample visual augmentation once so the two policies see the same images.
    teacher_observation = model.preprocess_observation(teacher_observation, train=True)
    teacher_observation = model._observation_to_dtype(
        teacher_observation, student.embed_dtype
    )
    visual = vision_observation(teacher_observation)
    batch_size = visual.state.shape[0]
    device = visual.state.device
    action_shape = (batch_size, student.action_horizon, student.action_dim)
    if noise is None:
        noise = torch.randn(action_shape, device=device, dtype=torch.float32)
    elif tuple(noise.shape) != action_shape:
        raise ValueError(
            f"Expected shared noise shape {action_shape}, got {tuple(noise.shape)}"
        )
    else:
        noise = noise.to(device=device, dtype=torch.float32)

    if query_index is None:
        query_index = int(torch.randint(num_steps, (), device=device).item())
    if type(query_index) is not int or not 0 <= query_index < num_steps:
        raise ValueError(f"query_index must be in [0, {num_steps})")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        index = torch.tensor(query_index, dtype=torch.long, device=device)
        # Equal prefix lengths keep FSDP collectives aligned across ranks.
        torch.distributed.broadcast(index, src=0)
        query_index = int(index.item())

    dt = -1.0 / num_steps
    with torch.no_grad():
        state = noise.detach()
        if query_index:
            cache, mask = _action_context(student, visual)
            for step in range(query_index):
                time = noise.new_full((batch_size,), 1.0 + step * dt)
                suffix = student.run_suffix(visual, state, time, cache, mask)
                state = state + dt * student.velocity_from_suffix(suffix).float()
            del cache, mask, suffix

        query_state = state.detach()
        query_time = noise.new_full((batch_size,), 1.0 + query_index * dt)
        teacher_velocity, teacher_features = _predict_at_query(
            teacher,
            teacher_observation,
            query_state,
            query_time,
            collect_features=collect_features,
        )

    student_velocity, student_features = _predict_at_query(
        student, visual, query_state, query_time, collect_features=collect_features
    )
    if (
        student_velocity.shape != teacher_velocity.shape
        or tuple(student_velocity.shape) != action_shape
    ):
        raise ValueError("Velocity matching requires full, aligned action tensors")
    action_loss = F.mse_loss(
        student_velocity.float(), teacher_velocity.detach().float()
    )
    if not collect_features:
        return action_loss, action_loss.new_zeros(())

    if not student_features or len(student_features) != len(teacher_features):
        raise ValueError("Teacher and student must expose corresponding action layers")
    layer_losses = []
    for student_layer, teacher_layer in zip(
        student_features, teacher_features, strict=True
    ):
        if student_layer.shape != teacher_layer.shape:
            raise ValueError("Corresponding action features must have equal shapes")
        layer_losses.append(
            F.mse_loss(student_layer.float(), teacher_layer.detach().float())
        )
    return action_loss, torch.stack(layer_losses).mean()
