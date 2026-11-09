"""Load Hugging Face ViT weights into the local shared architecture."""

from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vit-base-patch16-224-in21k"
    / "model.safetensors"
)


def _projection_parameter_prefix(
    model_state_keys: set[str],
    module_prefix: str,
) -> str:
    """Find whether a projection is a Linear or an adapter wrapping `.base`."""
    if f"{module_prefix}.weight" in model_state_keys:
        return module_prefix
    if f"{module_prefix}.base.weight" in model_state_keys:
        return f"{module_prefix}.base"
    raise KeyError(f"Cannot locate projection parameters: {module_prefix}")


def _detect_source_prefix(source_keys: set[str]) -> str:
    """Support both ViTModel and ViTForImageClassification checkpoints."""
    if "embeddings.cls_token" in source_keys:
        return ""
    if "vit.embeddings.cls_token" in source_keys:
        return "vit."
    raise RuntimeError("The checkpoint does not look like a Hugging Face ViT model.")


def build_vit_weight_map(
    model: nn.Module,
    source_prefix: str = "",
) -> dict[str, str]:
    """Map local backbone parameters to Hugging Face ViT parameter names."""
    if not hasattr(model, "config") or not hasattr(model.config, "num_layers"):
        raise TypeError("The model must expose a ViT config with num_layers.")

    model_state_keys = set(model.state_dict())
    weight_map = {
        "embedding.cls_token": f"{source_prefix}embeddings.cls_token",
        "embedding.position_embedding": f"{source_prefix}embeddings.position_embeddings",
        "embedding.patch_embedding.projection.weight": f"{source_prefix}embeddings.patch_embeddings.projection.weight",
        "embedding.patch_embedding.projection.bias": f"{source_prefix}embeddings.patch_embeddings.projection.bias",
        "head.norm.weight": f"{source_prefix}layernorm.weight",
        "head.norm.bias": f"{source_prefix}layernorm.bias",
    }

    for index in range(model.config.num_layers):
        ours = f"encoder.blocks.{index}"
        source = f"{source_prefix}encoder.layer.{index}"
        query = _projection_parameter_prefix(model_state_keys, f"{ours}.attention.query")
        key = _projection_parameter_prefix(model_state_keys, f"{ours}.attention.key")
        value = _projection_parameter_prefix(model_state_keys, f"{ours}.attention.value")
        output = _projection_parameter_prefix(model_state_keys, f"{ours}.attention.output")
        fc1 = _projection_parameter_prefix(model_state_keys, f"{ours}.mlp.fc1")
        fc2 = _projection_parameter_prefix(model_state_keys, f"{ours}.mlp.fc2")

        weight_map.update(
            {
                f"{ours}.norm1.weight": f"{source}.layernorm_before.weight",
                f"{ours}.norm1.bias": f"{source}.layernorm_before.bias",
                f"{query}.weight": f"{source}.attention.attention.query.weight",
                f"{query}.bias": f"{source}.attention.attention.query.bias",
                f"{key}.weight": f"{source}.attention.attention.key.weight",
                f"{key}.bias": f"{source}.attention.attention.key.bias",
                f"{value}.weight": f"{source}.attention.attention.value.weight",
                f"{value}.bias": f"{source}.attention.attention.value.bias",
                f"{output}.weight": f"{source}.attention.output.dense.weight",
                f"{output}.bias": f"{source}.attention.output.dense.bias",
                f"{ours}.norm2.weight": f"{source}.layernorm_after.weight",
                f"{ours}.norm2.bias": f"{source}.layernorm_after.bias",
                f"{fc1}.weight": f"{source}.intermediate.dense.weight",
                f"{fc1}.bias": f"{source}.intermediate.dense.bias",
                f"{fc2}.weight": f"{source}.output.dense.weight",
                f"{fc2}.bias": f"{source}.output.dense.bias",
            }
        )

    missing_targets = set(weight_map) - model_state_keys
    if missing_targets:
        raise RuntimeError(f"Weight map contains missing model keys: {missing_targets}")
    return weight_map


def load_pretrained_vit(
    model: nn.Module,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
) -> int:
    """Load the shared ViT backbone and return the tensor count."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Missing pretrained weights: {weights_path}\n"
            "Run `python -m scripts.download_pretrained_model` first."
        )

    source_state = load_file(weights_path, device="cpu")
    source_prefix = _detect_source_prefix(set(source_state))
    weight_map = build_vit_weight_map(model, source_prefix=source_prefix)

    missing_source_keys = set(weight_map.values()) - set(source_state)
    if missing_source_keys:
        raise RuntimeError(f"Checkpoint is missing parameters: {missing_source_keys}")

    mapped_state = {
        target_name: source_state[source_name]
        for target_name, source_name in weight_map.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state, strict=False)

    expected_missing = set(model.state_dict()) - set(mapped_state)
    if set(missing_keys) != expected_missing or unexpected_keys:
        raise RuntimeError(
            "Pretrained weights did not load as expected.\n"
            f"Missing: {missing_keys}\nUnexpected: {unexpected_keys}"
        )
    return len(mapped_state)
