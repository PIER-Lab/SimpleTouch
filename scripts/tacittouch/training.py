# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train the visual student with the existing UniVTAC FSDP pipeline."""

from __future__ import annotations

import copy
from argparse import Namespace
from typing import Any

import torch
from omegaconf import read_write

from rlinf.data.datasets.openpi_pytorch.univtac import build_univtac_sft_dataloader
from rlinf.models.embodiment.openpi_pytorch import resolve_pi0_config
from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker
from scripts.simpletouch.checkpoint import sha256_file
from scripts.tacittouch.objective import distillation_losses, vision_observation
from scripts.tacittouch.settings import TacitTouchSettings
from scripts.univtac.serve_policy import SimpleTacPolicy


def load_frozen_teacher(config, settings: TacitTouchSettings, device: str):
    marker = sha256_file(settings.teacher_marker)
    policy = SimpleTacPolicy(
        Namespace(
            device=device,
            checkpoint=str(settings.teacher_marker.parent),
            checkpoint_verify_mode="fast",
            task=str(config.data.task),
            norm_stats=str(config.data.norm_stats_path),
            tokenizer=str(config.data.tokenizer_path),
            tactile_encoder=settings.tactile_encoder_path,
            tactile_mode="true",
            num_steps=int(config.actor.model.num_steps),
        )
    )
    teacher = policy._model
    student_config = resolve_pi0_config(config.actor.model)
    if teacher.tactile_expert_wam_mode != "fast":
        raise ValueError("TacitTouch's teacher must use SimpleTouch's fast action path")
    for field in (
        "paligemma_variant",
        "action_expert_variant",
        "max_token_len",
        "discrete_state_input",
        "action_dim",
        "action_horizon",
    ):
        if getattr(student_config, field) != getattr(teacher._config, field):
            raise ValueError(f"Teacher and student differ in {field}")
    if policy._contract["env_action_dim"] != config.actor.model.action_dim:
        raise ValueError(
            "Teacher and student use different environment action dimensions"
        )
    if policy._contract["normalization"] != config.data.normalization:
        raise ValueError("Use the same normalization mode for teacher and student")
    if marker != sha256_file(settings.teacher_marker):
        raise RuntimeError("The teacher checkpoint changed during loading")
    return teacher


class TacitTouchActionModel(OpenPiPytorchSFTActionModel):
    """Same student parameters and state dict as native vision-only SFT."""

    def __init__(
        self,
        student,
        teacher,
        settings: TacitTouchSettings,
        *,
        num_steps,
        action_env_dim,
    ):
        super().__init__(student, num_steps=num_steps, action_env_dim=action_env_dim)
        self.settings = settings
        # The teacher is device-local and frozen. Do not register it as a child
        # module: only the student is sharded, optimized, and saved.
        object.__setattr__(self, "_teacher", teacher.eval().requires_grad_(False))

    def sft_forward(self, data: Any) -> dict[str, torch.Tensor]:
        observation, actions, action_valid_mask = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        action_loss, feature_loss = distillation_losses(
            self.model,
            self._teacher,
            observation,
            num_steps=self.num_steps,
            collect_features=self.settings.feature_loss_weight > 0,
        )
        loss = action_loss + self.settings.feature_loss_weight * feature_loss
        metrics = {
            "action_distillation_loss": action_loss.detach(),
            "feature_distillation_loss": feature_loss.detach(),
        }
        if self.settings.demonstration_loss_weight > 0:
            # Base policy flow-matching independently samples noise and time.
            demo_loss = self.model.compute_loss(
                vision_observation(observation),
                self._actions_to_device(actions),
                train=True,
            )
            demo_loss = self._masked_action_loss(demo_loss, action_valid_mask)
            loss = loss + self.settings.demonstration_loss_weight * demo_loss
            metrics["demonstration_loss"] = demo_loss.detach()
        return {"loss": loss, **metrics}


class TacitTouchSftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        # Reuse SimpleTouch's current-touch loader. This copy is solely for
        # dataset I/O, so the actual student config stays vision-only.
        data_config = copy.deepcopy(self.cfg)
        with read_write(data_config):
            data_config.actor.model.openpi.tactile_expert_enabled = True
            data_config.actor.model.openpi.tactile_encoder_path = (
                TacitTouchSettings.from_config(self.cfg).tactile_encoder_path
            )
        return build_univtac_sft_dataloader(
            data_config, self._world_size, self._rank, data_paths, eval_dataset
        )

    def model_provider_func(self):
        base = super().model_provider_func()
        settings = TacitTouchSettings.from_config(self.cfg)
        teacher = load_frozen_teacher(self.cfg, settings, device=f"cuda:{self.device}")
        return TacitTouchActionModel(
            base.model,
            teacher,
            settings,
            num_steps=base.num_steps,
            action_env_dim=base.action_env_dim,
        )
