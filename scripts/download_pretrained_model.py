"""Download the ImageNet-21k supervised ViT-B/16 backbone."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "google/vit-base-patch16-224-in21k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "pretrained" / "vit-base-patch16-224-in21k"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "README.md",
        ],
    )

    print(f"Model downloaded to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
