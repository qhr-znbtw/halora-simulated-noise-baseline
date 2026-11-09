"""Prepare memory-mapped VTAB-style dSprites orientation splits."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "dsprites"
    / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "dsprites" / "orientation_vtab"
RANDOM_SEED = 42
TRAIN_SIZE = 800
VAL_SIZE = 200


def orientation_labels(latent_classes: np.ndarray) -> np.ndarray:
    """Group the 40 original angles into the 16 classes used by VTAB."""
    original = latent_classes[:, 3].astype(np.float32)
    return np.floor(original / (40.0 / 16.0)).astype(np.int64)


def split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build balanced 800/200 subsets and a stratified 10% test set."""
    generator = np.random.default_rng(RANDOM_SEED)
    train_pool: list[int] = []
    val_pool: list[int] = []
    test_indices: list[int] = []

    for class_id in range(16):
        indices = np.flatnonzero(labels == class_id)
        generator.shuffle(indices)
        test_size = round(0.1 * len(indices))
        val_pool_size = round(0.1 * len(indices))
        test_indices.extend(indices[:test_size].tolist())
        val_pool.extend(indices[test_size : test_size + val_pool_size].tolist())
        train_pool.extend(indices[test_size + val_pool_size :].tolist())

    def balanced_subset(pool: list[int], size: int) -> np.ndarray:
        pool_array = np.asarray(pool, dtype=np.int64)
        base, remainder = divmod(size, 16)
        chosen: list[int] = []
        for class_id in range(16):
            candidates = pool_array[labels[pool_array] == class_id]
            generator.shuffle(candidates)
            class_size = base + int(class_id < remainder)
            chosen.extend(candidates[:class_size].tolist())
        generator.shuffle(chosen)
        return np.asarray(chosen, dtype=np.int64)

    train_indices = balanced_subset(train_pool, TRAIN_SIZE)
    val_indices = balanced_subset(val_pool, VAL_SIZE)
    test_array = np.asarray(test_indices, dtype=np.int64)
    generator.shuffle(test_array)
    return train_indices, val_indices, test_array


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing raw dSprites archive: {SOURCE_PATH}\n"
            "Run `python -m scripts.download_mini_vtab` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading the compressed dSprites archive once. This uses about 3 GB RAM.")
    with np.load(SOURCE_PATH, allow_pickle=True) as archive:
        labels = orientation_labels(archive["latents_classes"])
        train_indices, val_indices, test_indices = split_indices(labels)
        images = archive["imgs"]

        for split, indices in (
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ):
            np.save(OUTPUT_DIR / f"{split}_images.npy", images[indices])
            np.save(OUTPUT_DIR / f"{split}_labels.npy", labels[indices])
            print(f"Prepared {split}: {len(indices)} examples")

    print(f"Prepared dSprites data written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
