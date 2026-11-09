"""Evaluation loop shared by all transfer-learning experiments."""

import torch
from torch import nn
from torch.utils.data import DataLoader

def test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    """评估模型，返回平均loss、accuracy、正确数和样本数。"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with torch.inference_mode():
        for images, labels in dataloader:
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
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("评估DataLoader中没有样本。")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "correct": total_correct,
        "total": total_samples,
    }
