"""ViT-B/16 with paper-aligned HaLoRA targets and RRAM noise controls."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from experiments.halora.config import LoRAConfig, NoiseConfig
from experiments.halora.layers import HaLoRALinear, NoisyLinear
from vit_core.vit_backbone import (
    VIT_BASE_PATCH16_224,
    ViTConfig,
    ViTForImageClassification,
)


class ViTHaLoRAClassifier(ViTForImageClassification):
    """Apply LoRA to Q/K/V/fc1/fc2 and RRAM noise to all encoder linears."""

    def __init__(
        self,
        num_classes: int,
        *,
        lora: LoRAConfig,
        noise: NoiseConfig,
        vit_config: ViTConfig = VIT_BASE_PATCH16_224,
    ) -> None:
        target_factory = partial(
            HaLoRALinear,
            rank=lora.rank,
            alpha=lora.alpha,
            sigma=noise.sigma,
            block_size=noise.block_size,
        )
        noisy_backbone_factory = partial(
            NoisyLinear,
            sigma=noise.sigma,
            block_size=noise.block_size,
        )
        super().__init__(
            num_classes=num_classes,
            config=vit_config,
            query_factory=target_factory,
            key_factory=target_factory,
            value_factory=target_factory,
            output_factory=noisy_backbone_factory,
            fc1_factory=target_factory,
            fc2_factory=target_factory,
        )
        self.set_trainable_parameters()

    def lora_modules(self) -> list[HaLoRALinear]:
        return [module for module in self.modules() if isinstance(module, HaLoRALinear)]

    def noisy_modules(self) -> list[NoisyLinear]:
        return [module for module in self.modules() if isinstance(module, NoisyLinear)]

    def set_trainable_parameters(self) -> None:
        """Expose only LoRA factors and the task-specific classifier."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.lora_modules():
            for parameter in module.lora_down.parameters():
                parameter.requires_grad = True
            for parameter in module.lora_up.parameters():
                parameter.requires_grad = True
        for parameter in self.head.classifier.parameters():
            parameter.requires_grad = True

    def regularization_loss(self) -> torch.Tensor:
        modules = self.lora_modules()
        if not modules:
            raise RuntimeError("No LoRA modules are present")
        return torch.stack([module.regularization_loss() for module in modules]).sum()

    def enable_noise(self) -> None:
        for module in self.noisy_modules():
            module.enable_noise()

    def disable_noise(self) -> None:
        for module in self.noisy_modules():
            module.disable_noise()

    @torch.no_grad()
    def resample_noise(self, generator: torch.Generator) -> None:
        for module in self.noisy_modules():
            module.resample_noise(generator)

    @torch.no_grad()
    def resample_noise_from_seed(self, seed: int) -> None:
        device = next(self.parameters()).device
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        self.resample_noise(generator)
