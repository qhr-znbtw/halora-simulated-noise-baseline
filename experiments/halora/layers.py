"""Noise-aware frozen linear layers and LoRA branches."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def blockwise_gaussian_noise(
    weight: torch.Tensor,
    *,
    sigma: float,
    block_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample elementwise Gaussian noise scaled by each block's max |weight|."""
    if weight.ndim != 2:
        raise ValueError("RRAM noise is defined only for two-dimensional weights")
    if sigma <= 0 or block_size <= 0:
        raise ValueError("sigma and block_size must be positive")

    rows, columns = weight.shape
    padded_rows = math.ceil(rows / block_size) * block_size
    padded_columns = math.ceil(columns / block_size) * block_size
    pad_rows = padded_rows - rows
    pad_columns = padded_columns - columns

    clean = weight.detach().to(dtype=torch.float32)
    padded = F.pad(clean, (0, pad_columns, 0, pad_rows))
    row_blocks = padded_rows // block_size
    column_blocks = padded_columns // block_size
    blocks = (
        padded.reshape(row_blocks, block_size, column_blocks, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    scales = blocks.abs().amax(dim=(-2, -1), keepdim=True)
    standard_normal = torch.randn(
        blocks.shape,
        generator=generator,
        device=weight.device,
        dtype=torch.float32,
    )
    noisy_blocks = standard_normal.mul_(sigma).mul_(scales)
    noise = (
        noisy_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(padded_rows, padded_columns)
    )
    return noise[:rows, :columns].to(dtype=weight.dtype)


class NoisyLinear(nn.Module):
    """Frozen-compatible linear layer with replaceable, non-accumulating RRAM noise."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        sigma: float,
        block_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.block_size = int(block_size)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer(
            "_rram_noise",
            torch.zeros_like(self.base.weight),
            persistent=False,
        )
        self._noise_enabled = False

    @property
    def noise_enabled(self) -> bool:
        return self._noise_enabled

    def enable_noise(self) -> None:
        self._noise_enabled = True

    def disable_noise(self) -> None:
        self._noise_enabled = False

    @torch.no_grad()
    def resample_noise(self, generator: torch.Generator) -> None:
        sampled = blockwise_gaussian_noise(
            self.base.weight,
            sigma=self.sigma,
            block_size=self.block_size,
            generator=generator,
        )
        self._rram_noise.copy_(sampled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.base.weight
        if self._noise_enabled:
            weight = weight + self._rram_noise
        return F.linear(inputs, weight, self.base.bias)


class HaLoRALinear(NoisyLinear):
    """Frozen noisy backbone projection plus an accurate trainable LoRA branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        sigma: float,
        block_size: int,
        bias: bool = True,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        super().__init__(
            in_features,
            out_features,
            sigma=sigma,
            block_size=block_size,
            bias=bias,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def regularization_loss(self) -> torch.Tensor:
        """Return the paper's unsquared Euclidean/Frobenius Gram penalty."""
        down = self.lora_down.weight.float()
        up = self.lora_up.weight.float()
        down_small_gram = down @ down.transpose(0, 1)
        up_small_gram = up.transpose(0, 1) @ up
        return torch.linalg.matrix_norm(
            down_small_gram, ord="fro"
        ) + torch.linalg.matrix_norm(up_small_gram, ord="fro")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        backbone = super().forward(inputs)
        update = self.lora_up(self.lora_down(inputs))
        return backbone + self.scaling * update
