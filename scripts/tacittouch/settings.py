# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Simulation settings kept separate from the deployed policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from scripts.simpletouch.checkpoint import sha256_file


@dataclass(frozen=True)
class TacitTouchSettings:
    teacher_checkpoint: str
    tactile_encoder_path: str
    feature_loss_weight: float = 0.01
    demonstration_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("teacher_checkpoint", "tactile_encoder_path"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"Set actor.model.tacittouch.{name}")
        for name in ("feature_loss_weight", "demonstration_loss_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @classmethod
    def from_config(cls, config) -> TacitTouchSettings:
        return cls(
            **OmegaConf.to_container(config.actor.model.tacittouch, resolve=True)
        )

    @property
    def teacher_marker(self) -> Path:
        return Path(self.teacher_checkpoint).expanduser().resolve() / "_SUCCESS"


def validate_training_config(config) -> None:
    model_config = config.actor.model
    openpi = model_config.openpi
    if model_config.model_type != "openpi_pytorch" or openpi.get("task") != "sft":
        raise ValueError("TacitTouch uses the native PyTorch UniVTAC SFT pipeline")
    if openpi.tactile_expert_enabled or openpi.tactile_expert_future_loss_weight:
        raise ValueError("TacitTouch's deployed student must be vision-only")
    if model_config.num_steps != 10 or model_config.num_action_chunks != 50:
        raise ValueError(
            "The TacitTouch recipe uses 10 Euler steps and H50 action chunks"
        )
    if config.runner.val_check_interval != -1 or config.data.get("val_data_paths"):
        raise ValueError("Use the UniVTAC evaluator for validation after training")
    if config.data.get("tactile_world_cache_path") or config.data.get(
        "tactile_expert_future_cache_path"
    ):
        raise ValueError(
            "TacitTouch reads current measured touch from the demonstrations"
        )
    settings = TacitTouchSettings.from_config(config)
    if not settings.teacher_marker.is_file():
        raise FileNotFoundError(
            f"Pass a completed SimpleTouch teacher checkpoint directory: {settings.teacher_marker}"
        )


def training_provenance(config) -> dict:
    """Record the objective and teacher identity for student checkpoint resumption."""
    settings = TacitTouchSettings.from_config(config)
    return {
        "teacher_marker_sha256": sha256_file(settings.teacher_marker),
        "num_denoising_steps": int(config.actor.model.num_steps),
        "feature_loss_weight": settings.feature_loss_weight,
        "demonstration_loss_weight": settings.demonstration_loss_weight,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("objective.py", "settings.py", "training.py")
        },
    }
