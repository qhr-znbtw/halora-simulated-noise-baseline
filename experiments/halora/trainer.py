"""Training loop with independent noise-refresh and regularization schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from experiments.halora.config import ExperimentConfig
from experiments.halora.evaluation import evaluate
from experiments.halora.model import ViTHaLoRAClassifier
from experiments.halora.schedule import (
    should_apply_regularizer,
    should_refresh_noise,
)
from vit_core.checkpoint import load_trainable_checkpoint, save_trainable_checkpoint


def _noise_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def train_one_epoch(
    *,
    model: ViTHaLoRAClassifier,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    global_step: int,
    noise_generator: torch.Generator,
    max_batches: int | None = None,
) -> tuple[dict[str, float | int], int]:
    model.train()
    if config.condition.training_noise:
        model.enable_noise()
    else:
        model.disable_noise()

    total_task_loss = 0.0
    total_regularizer_loss = 0.0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    noise_refreshes = 0
    regularizer_applications = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if config.condition.training_noise and should_refresh_noise(
            global_step, config.noise.refresh_steps
        ):
            model.resample_noise(noise_generator)
            noise_refreshes += 1

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(images)
            task_loss = criterion(logits, labels)

        regularizer_loss = torch.zeros((), device=device, dtype=torch.float32)
        if config.condition.regularizer and should_apply_regularizer(
            global_step, config.regularizer.interval
        ):
            regularizer_loss = model.regularization_loss()
            regularizer_applications += 1

        loss = task_loss.float() + config.regularizer.mu * regularizer_loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_task_loss += float(task_loss.detach().item()) * batch_size
        total_regularizer_loss += float(regularizer_loss.detach().item()) * batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size
        global_step += 1

    if total_samples == 0:
        raise ValueError("Training DataLoader produced no samples")
    return (
        {
            "task_loss": total_task_loss / total_samples,
            "regularizer_loss": total_regularizer_loss / total_samples,
            "total_loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "samples": total_samples,
            "noise_refreshes": noise_refreshes,
            "regularizer_applications": regularizer_applications,
        },
        global_step,
    )


def train_model(
    *,
    model: ViTHaLoRAClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    criterion: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    num_epochs: int,
    checkpoint_path: str | Path,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Train fixed epochs, select by clean validation accuracy, restore the best."""
    model.to(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    generator = _noise_generator(device, config.noise.train_seed)
    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    global_step = 0
    total_noise_refreshes = 0
    total_regularizer_applications = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            global_step=global_step,
            noise_generator=generator,
            max_batches=max_train_batches,
        )
        total_noise_refreshes += int(train_metrics["noise_refreshes"])
        total_regularizer_applications += int(
            train_metrics["regularizer_applications"]
        )

        model.disable_noise()
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_eval_batches,
        )
        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation_clean": val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | step {global_step:05d} | "
            f"lr {learning_rate:.2e} | task {train_metrics['task_loss']:.4f} | "
            f"reg {train_metrics['regularizer_loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | "
            f"clean val acc {val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            save_trainable_checkpoint(
                model,
                checkpoint_path,
                epoch=epoch,
                val_accuracy=best_val_accuracy,
                extra={
                    "config": config.as_dict(),
                    "global_step": global_step,
                    "checkpoint_selection": "best_clean_validation_accuracy",
                },
            )
        if scheduler is not None:
            scheduler.step()

    model.disable_noise()
    best_checkpoint = load_trainable_checkpoint(model, checkpoint_path)
    model.to(device)
    summary = {
        "global_steps": global_step,
        "noise_refreshes": total_noise_refreshes,
        "regularizer_applications": total_regularizer_applications,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_clean_val_accuracy": float(best_checkpoint["val_accuracy"]),
    }
    return history, summary
