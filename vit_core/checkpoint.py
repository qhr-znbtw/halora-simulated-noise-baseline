
"""Checkpoint helpers shared by all transfer-learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU copies of exactly the parameters optimized in this run."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_trainable_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    val_accuracy: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save trainable parameters and enough metadata to validate restoration."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = trainable_state_dict(model)
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "trainable_only": True,
        "trainable_parameter_names": sorted(state),
        "model_state_dict": state,
    }
    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, checkpoint_path)


def load_trainable_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Restore a trainable-only checkpoint into an initialized base model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("checkpoint中缺少model_state_dict。")

    saved_state = checkpoint["model_state_dict"]
    expected_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved_state) != expected_names:
        missing = sorted(expected_names - set(saved_state))
        unexpected = sorted(set(saved_state) - expected_names)
        raise RuntimeError(
            "checkpoint中的参数与当前可训练参数不一致。\n"
            f"缺少：{missing}\n"
            f"多余：{unexpected}"
        )

    model.load_state_dict(saved_state, strict=False)
    return checkpoint
