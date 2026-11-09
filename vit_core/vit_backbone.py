
"""Reusable Vision Transformer components shared by all experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


LinearFactory = Callable[[int, int], nn.Module]


@dataclass(frozen=True)
class ViTConfig:
    """Structural hyperparameters of a Vision Transformer."""

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072
    num_layers: int = 12
    layer_norm_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size必须能被patch_size整除。")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size必须能被num_heads整除。")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


VIT_BASE_PATCH16_224 = ViTConfig()
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def build_image_transform(
    train: bool,
    image_size: int = VIT_BASE_PATCH16_224.image_size,
) -> transforms.Compose:
    """Build the training or evaluation preprocessing pipeline."""
    if train:
        spatial_transforms = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        spatial_transforms = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
        ]

    return transforms.Compose(
        spatial_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


class PatchEmbedding(nn.Module):
    """Convert an image batch into a sequence of patch tokens."""

    def __init__(self, config: ViTConfig = VIT_BASE_PATCH16_224) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("ViT输入必须具有[B, C, H, W]四个维度。")
        if images.shape[1:] != (
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError(
                "输入形状错误：期望每张图像为"
                f"[{self.config.num_channels}, {self.config.image_size}, "
                f"{self.config.image_size}]，实际为{list(images.shape[1:])}。"
            )

        tokens = self.projection(images)
        return tokens.flatten(2).transpose(1, 2)


class ViTInputEmbedding(nn.Module):
    """Prepend the CLS token and add learned positional embeddings."""

    def __init__(self, config: ViTConfig = VIT_BASE_PATCH16_224) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.hidden_size)
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embedding(images)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        return tokens + self.position_embedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with replaceable projection modules."""

    def __init__(
        self,
        config: ViTConfig = VIT_BASE_PATCH16_224,
        query_factory: LinearFactory = nn.Linear,
        key_factory: LinearFactory = nn.Linear,
        value_factory: LinearFactory = nn.Linear,
        output_factory: LinearFactory = nn.Linear,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.hidden_size // config.num_heads

        self.query = query_factory(config.hidden_size, config.hidden_size)
        self.key = key_factory(config.hidden_size, config.hidden_size)
        self.value = value_factory(config.hidden_size, config.hidden_size)
        self.output = output_factory(config.hidden_size, config.hidden_size)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, hidden_size = tensor.shape
        expected_hidden_size = self.num_heads * self.head_size
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"token维度应为{expected_hidden_size}，实际为{hidden_size}。"
            )
        tensor = tensor.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_size,
        )
        return tensor.transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(tokens))
        value = self._split_heads(self.value(tokens))

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_weights = attention_scores.softmax(dim=-1)

        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().flatten(2)
        return self.output(context)


class MLP(nn.Module):
    """The token-wise feed-forward network in a ViT encoder block."""

    def __init__(
        self,
        config: ViTConfig = VIT_BASE_PATCH16_224,
        fc1_factory: LinearFactory = nn.Linear,
        fc2_factory: LinearFactory = nn.Linear,
    ) -> None:
        super().__init__()
        self.fc1 = fc1_factory(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = fc2_factory(config.intermediate_size, config.hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(tokens)))


class TransformerEncoderBlock(nn.Module):
    """A pre-LayerNorm Vision Transformer encoder block."""

    def __init__(
        self,
        config: ViTConfig = VIT_BASE_PATCH16_224,
        query_factory: LinearFactory = nn.Linear,
        key_factory: LinearFactory = nn.Linear,
        value_factory: LinearFactory = nn.Linear,
        output_factory: LinearFactory = nn.Linear,
        fc1_factory: LinearFactory = nn.Linear,
        fc2_factory: LinearFactory = nn.Linear,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MultiHeadSelfAttention(
            config=config,
            query_factory=query_factory,
            key_factory=key_factory,
            value_factory=value_factory,
            output_factory=output_factory,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(
            config,
            fc1_factory=fc1_factory,
            fc2_factory=fc2_factory,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class TransformerEncoder(nn.Module):
    """A stack of Vision Transformer encoder blocks."""

    def __init__(
        self,
        config: ViTConfig = VIT_BASE_PATCH16_224,
        query_factory: LinearFactory = nn.Linear,
        key_factory: LinearFactory = nn.Linear,
        value_factory: LinearFactory = nn.Linear,
        output_factory: LinearFactory = nn.Linear,
        fc1_factory: LinearFactory = nn.Linear,
        fc2_factory: LinearFactory = nn.Linear,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config=config,
                    query_factory=query_factory,
                    key_factory=key_factory,
                    value_factory=value_factory,
                    output_factory=output_factory,
                    fc1_factory=fc1_factory,
                    fc2_factory=fc2_factory,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class ClassificationHead(nn.Module):
    """Normalize the final CLS representation and compute class logits."""

    def __init__(
        self,
        num_classes: int,
        config: ViTConfig = VIT_BASE_PATCH16_224,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized_tokens = self.norm(tokens)
        return self.classifier(normalized_tokens[:, 0])


class ViTForImageClassification(nn.Module):
    """A configurable ViT classifier used by every transfer-learning method."""

    def __init__(
        self,
        num_classes: int,
        config: ViTConfig = VIT_BASE_PATCH16_224,
        query_factory: LinearFactory = nn.Linear,
        key_factory: LinearFactory = nn.Linear,
        value_factory: LinearFactory = nn.Linear,
        output_factory: LinearFactory = nn.Linear,
        fc1_factory: LinearFactory = nn.Linear,
        fc2_factory: LinearFactory = nn.Linear,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = ViTInputEmbedding(config)
        self.encoder = TransformerEncoder(
            config=config,
            query_factory=query_factory,
            key_factory=key_factory,
            value_factory=value_factory,
            output_factory=output_factory,
            fc1_factory=fc1_factory,
            fc2_factory=fc2_factory,
        )
        self.head = ClassificationHead(num_classes=num_classes, config=config)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the final sequence of encoded tokens."""
        return self.encoder(self.embedding(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))
