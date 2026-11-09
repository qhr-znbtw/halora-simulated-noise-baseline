"""Clean and fixed-hardware-noise evaluation for HaLoRA experiments."""

from __future__ import annotations

import statistics
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.halora.config import EvaluationConfig
from experiments.halora.model import ViTHaLoRAClassifier


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate a model under its currently active hardware-noise state."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "total": total_samples,
    }


def evaluate_clean_and_noisy(
    model: ViTHaLoRAClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    evaluation: EvaluationConfig,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate clean once and noisy once per fixed noise realization seed."""
    model.disable_noise()
    clean = evaluate(
        model,
        dataloader,
        criterion,
        device,
        max_batches=max_batches,
    )

    noisy_runs: list[dict[str, float | int]] = []
    for seed in evaluation.noise_seeds:
        model.resample_noise_from_seed(seed)
        model.enable_noise()
        metrics = evaluate(
            model,
            dataloader,
            criterion,
            device,
            max_batches=max_batches,
        )
        noisy_runs.append({"seed": seed, **metrics})

    model.disable_noise()
    noisy_accuracies = [float(run["accuracy"]) for run in noisy_runs]
    noisy_losses = [float(run["loss"]) for run in noisy_runs]
    return {
        "clean": clean,
        "noise_sigma": evaluation.noise_sigma,
        "noisy_runs": noisy_runs,
        "noisy_summary": {
            "accuracy_mean": statistics.fmean(noisy_accuracies),
            "accuracy_population_std": statistics.pstdev(noisy_accuracies),
            "loss_mean": statistics.fmean(noisy_losses),
            "loss_population_std": statistics.pstdev(noisy_losses),
            "num_noise_realizations": len(noisy_runs),
        },
    }
