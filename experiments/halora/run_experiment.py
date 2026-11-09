"""Train and evaluate one controlled HaLoRA condition on one Mini-VTAB task."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from experiments.halora.config import ExperimentConfig, load_experiment_config
from experiments.halora.evaluation import evaluate_clean_and_noisy
from experiments.halora.model import ViTHaLoRAClassifier
from experiments.halora.trainer import train_model
from vit_core.mini_vtab import TASK_SPECS, build_dataloaders, get_task_spec
from vit_core.pretrained import load_pretrained_vit


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> AdamW:
    lora_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_down." in name or ".lora_up." in name:
            lora_parameters.append(parameter)
        elif name.startswith("head.classifier."):
            classifier_parameters.append(parameter)
        else:
            raise RuntimeError(f"Unclassified trainable parameter: {name}")
    if not lora_parameters or not classifier_parameters:
        raise RuntimeError("LoRA and classifier parameter groups must both be non-empty")
    return AdamW(
        [
            {
                "params": lora_parameters,
                "lr": config.training.lora_learning_rate,
            },
            {
                "params": classifier_parameters,
                "lr": config.training.classifier_learning_rate,
            },
        ],
        weight_decay=config.training.weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    num_epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    if not 0 <= warmup_epochs < num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs)")

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        decay_epochs = num_epochs - warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(decay_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_down." in name or ".lora_up." in name
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora": lora,
        "trainable_fraction": trainable / total,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    config = load_experiment_config(config_path)
    task = get_task_spec(args.task)
    set_random_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protocol = "smoke" if args.smoke else "development_800_200"
    result_path = (
        PROJECT_ROOT
        / "results"
        / config.condition.name
        / protocol
        / task.name
        / f"seed_{config.training.seed}.json"
    )
    checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / config.condition.name
        / protocol
        / task.name
        / f"seed_{config.training.seed}.pt"
    )
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Result already exists: {result_path}. Pass --overwrite to replace it."
        )

    num_workers = 0 if args.smoke else args.num_workers
    train_loader, val_loader, test_loader = build_dataloaders(
        task.name,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_workers=num_workers,
    )
    model = ViTHaLoRAClassifier(
        num_classes=task.num_classes,
        lora=config.lora,
        noise=config.noise,
    )
    loaded_tensors = load_pretrained_vit(model)
    model.set_trainable_parameters()
    optimizer = build_optimizer(model, config)

    num_epochs = 1 if args.smoke else config.training.epochs
    warmup_epochs = 0 if args.smoke else config.training.warmup_epochs
    scheduler = build_scheduler(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    parameter_counts = count_parameters(model)
    max_train_batches = 2 if args.smoke else None
    max_eval_batches = 2 if args.smoke else None

    print(f"Condition: {config.condition.name}", flush=True)
    print(f"Task: {task.name} ({task.group}, {task.num_classes} classes)", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        f"LoRA: targets={config.lora.targets}, rank={config.lora.rank}, "
        f"alpha={config.lora.alpha:g}, modules={len(model.lora_modules())}",
        flush=True,
    )
    print(
        f"Noise: train={config.condition.training_noise}, sigma={config.noise.sigma:g}, "
        f"block={config.noise.block_size}, refresh={config.noise.refresh_steps}, "
        f"noisy modules={len(model.noisy_modules())}",
        flush=True,
    )
    print(
        f"Regularizer: enabled={config.condition.regularizer}, "
        f"mu={config.regularizer.mu:g}, interval={config.regularizer.interval}",
        flush=True,
    )
    print(f"Loaded pretrained tensors: {loaded_tensors}", flush=True)
    print(f"Parameters: {parameter_counts}", flush=True)

    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_start = time.perf_counter()
    history, training_summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    evaluation_start = time.perf_counter()
    test_metrics = evaluate_clean_and_noisy(
        model,
        test_loader,
        criterion,
        device,
        config.evaluation,
        max_batches=max_eval_batches,
    )
    synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    result: dict[str, Any] = {
        "protocol": protocol,
        "condition": config.condition.name,
        "task": task.name,
        "task_group": task.group,
        "num_classes": task.num_classes,
        "config_path": str(config_path),
        "config": config.as_dict(),
        "dataset_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "loaded_pretrained_tensors": loaded_tensors,
        "lora_module_count": len(model.lora_modules()),
        "noisy_module_count": len(model.noisy_modules()),
        "parameters": parameter_counts,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "training_summary": training_summary,
        "test": test_metrics,
        "history": history,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    temporary_path.replace(result_path)

    print("Final test result", flush=True)
    print(
        f"Clean accuracy: {test_metrics['clean']['accuracy']:.4f} | "
        f"Noisy accuracy: {test_metrics['noisy_summary']['accuracy_mean']:.4f} "
        f"+/- {test_metrics['noisy_summary']['accuracy_population_std']:.4f}",
        flush=True,
    )
    print(f"Result file: {result_path}", flush=True)


if __name__ == "__main__":
    main()
