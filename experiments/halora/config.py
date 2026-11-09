"""Typed configuration for the four controlled HaLoRA experiment conditions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


EXPECTED_TARGETS = ("query", "key", "value", "fc1", "fc2")


@dataclass(frozen=True)
class ConditionConfig:
    name: str
    training_noise: bool
    regularizer: bool


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    targets: tuple[str, ...]


@dataclass(frozen=True)
class NoiseConfig:
    sigma: float
    block_size: int
    refresh_steps: int
    train_seed: int
    granularity: str
    scope: str


@dataclass(frozen=True)
class RegularizerConfig:
    mu: float
    interval: int
    norm: str
    reduction: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    warmup_epochs: int
    train_batch_size: int
    eval_batch_size: int
    lora_learning_rate: float
    classifier_learning_rate: float
    weight_decay: float
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    clean: bool
    noise_sigma: float
    noise_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    condition: ConditionConfig
    lora: LoRAConfig
    noise: NoiseConfig
    regularizer: RegularizerConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def validate(self) -> None:
        if not self.condition.name:
            raise ValueError("condition.name cannot be empty")
        if self.lora.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.lora.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if self.lora.targets != EXPECTED_TARGETS:
            raise ValueError(
                f"LoRA targets must be exactly {EXPECTED_TARGETS}, got {self.lora.targets}"
            )
        if self.noise.sigma <= 0:
            raise ValueError("noise sigma must be positive")
        if self.noise.block_size <= 0 or self.noise.refresh_steps <= 0:
            raise ValueError("noise block size and refresh interval must be positive")
        if self.noise.granularity != "elementwise_iid":
            raise ValueError("only elementwise_iid block noise is supported")
        if self.noise.scope != "encoder_linear_weights":
            raise ValueError("noise scope must be encoder_linear_weights")
        if self.regularizer.mu < 0 or self.regularizer.interval <= 0:
            raise ValueError("regularizer mu must be non-negative and interval positive")
        if self.regularizer.norm != "frobenius_unsquared":
            raise ValueError("regularizer norm must be frobenius_unsquared")
        if self.regularizer.reduction != "sum":
            raise ValueError("regularizer reduction must be sum")
        if (
            self.condition.regularizer
            and self.regularizer.interval != self.noise.refresh_steps
        ):
            raise ValueError(
                "the strict HaLoRA protocol applies regularization only when "
                "the hardware-noise map is refreshed"
            )
        if self.training.epochs <= 0:
            raise ValueError("training epochs must be positive")
        if not 0 <= self.training.warmup_epochs < self.training.epochs:
            raise ValueError("warmup epochs must be in [0, epochs)")
        if self.training.train_batch_size <= 0 or self.training.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.evaluation.noise_sigma != self.noise.sigma:
            raise ValueError("the initial experiment uses one shared train/test sigma")
        if not self.evaluation.noise_seeds:
            raise ValueError("at least one evaluation noise seed is required")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    condition = ConditionConfig(**raw["condition"])
    lora_raw = dict(raw["lora"])
    lora_raw["targets"] = tuple(lora_raw["targets"])
    evaluation_raw = dict(raw["evaluation"])
    evaluation_raw["noise_seeds"] = tuple(evaluation_raw["noise_seeds"])

    config = ExperimentConfig(
        condition=condition,
        lora=LoRAConfig(**lora_raw),
        noise=NoiseConfig(**raw["noise"]),
        regularizer=RegularizerConfig(**raw["regularizer"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**evaluation_raw),
    )
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return _build_config(raw)
