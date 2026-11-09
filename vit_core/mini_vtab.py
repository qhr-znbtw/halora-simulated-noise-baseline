"""Datasets and fixed splits for the custom three-task Mini-VTAB suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import DTD, EuroSAT
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_SEED = 42
TRAIN_SIZE = 800
VAL_SIZE = 200
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 4


@dataclass(frozen=True)
class TaskSpec:
    name: str
    num_classes: int
    group: str


TASK_SPECS = {
    "dtd": TaskSpec(name="dtd", num_classes=47, group="natural"),
    "eurosat": TaskSpec(name="eurosat", num_classes=10, group="specialized"),
    "dsprites_orientation": TaskSpec(
        name="dsprites_orientation",
        num_classes=16,
        group="structured",
    ),
}


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as error:
        choices = ", ".join(TASK_SPECS)
        raise ValueError(f"Unknown task {task_name!r}. Choose one of: {choices}.") from error


def build_image_transform() -> transforms.Compose:
    """Use one task-independent VTAB preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def _balanced_sample(
    labels: list[int] | np.ndarray,
    sample_size: int,
    *,
    seed: int,
) -> list[int]:
    """Choose a deterministic near-class-balanced subset of local indices."""
    labels_array = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels_array)
    if sample_size > labels_array.size:
        raise ValueError("sample_size cannot exceed the source dataset size.")

    base, remainder = divmod(sample_size, len(classes))
    generator = np.random.default_rng(seed)
    chosen: list[int] = []

    for position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(labels_array == class_id)
        generator.shuffle(class_indices)
        class_size = base + int(position < remainder)
        if class_size > class_indices.size:
            raise ValueError(f"Class {class_id} has too few samples.")
        chosen.extend(class_indices[:class_size].tolist())

    generator.shuffle(chosen)
    return chosen


def _stratified_three_way_split(
    labels: list[int] | np.ndarray,
    *,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic 60/20/20 train/validation/test pools."""
    labels_array = np.asarray(labels, dtype=np.int64)
    generator = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == class_id)
        generator.shuffle(class_indices)
        train_end = round(0.6 * class_indices.size)
        val_end = train_end + round(0.2 * class_indices.size)
        train_indices.extend(class_indices[:train_end].tolist())
        val_indices.extend(class_indices[train_end:val_end].tolist())
        test_indices.extend(class_indices[val_end:].tolist())

    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    generator.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def _build_dtd_datasets(data_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    transform = build_image_transform()
    root = data_dir / "dtd"
    train_source = DTD(root=root, split="train", transform=transform, download=False)
    val_source = DTD(root=root, split="val", transform=transform, download=False)
    test_dataset = DTD(root=root, split="test", transform=transform, download=False)

    train_indices = _balanced_sample(
        train_source._labels,
        TRAIN_SIZE,
        seed=RANDOM_SEED,
    )
    val_indices = _balanced_sample(
        val_source._labels,
        VAL_SIZE,
        seed=RANDOM_SEED + 1,
    )
    return Subset(train_source, train_indices), Subset(val_source, val_indices), test_dataset


def _build_eurosat_datasets(data_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    transform = build_image_transform()
    root = data_dir / "eurosat"
    source = EuroSAT(root=root, transform=transform, download=False)
    train_pool, val_pool, test_indices = _stratified_three_way_split(
        source.targets,
        seed=RANDOM_SEED,
    )

    train_local = _balanced_sample(
        [source.targets[index] for index in train_pool],
        TRAIN_SIZE,
        seed=RANDOM_SEED + 2,
    )
    val_local = _balanced_sample(
        [source.targets[index] for index in val_pool],
        VAL_SIZE,
        seed=RANDOM_SEED + 3,
    )
    train_indices = [train_pool[index] for index in train_local]
    val_indices = [val_pool[index] for index in val_local]
    return Subset(source, train_indices), Subset(source, val_indices), Subset(source, test_indices)


class DSpritesOrientationDataset(Dataset):
    """Memory-mapped prepared dSprites images and 16 orientation classes."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.split = split
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        if len(self.images) != len(self.labels):
            raise RuntimeError(f"dSprites {split} images and labels differ in length.")
        self.transform = build_image_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_array = np.asarray(self.images[index], dtype=np.uint8) * 255
        image = Image.fromarray(image_array, mode="L").convert("RGB")
        return self.transform(image), int(self.labels[index])


def _prepared_dsprites_root(data_dir: Path) -> Path:
    return data_dir / "dsprites" / "orientation_vtab"


def _build_dsprites_datasets(data_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    root = _prepared_dsprites_root(data_dir)
    required = [
        root / f"{split}_{kind}.npy"
        for split in ("train", "val", "test")
        for kind in ("images", "labels")
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared dSprites files are missing. Run "
            "`python -m scripts.prepare_dsprites_orientation` first."
        )
    return (
        DSpritesOrientationDataset(root, "train"),
        DSpritesOrientationDataset(root, "val"),
        DSpritesOrientationDataset(root, "test"),
    )


def build_datasets(
    task_name: str,
    data_dir: str | Path = DATA_DIR,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return the fixed train, validation, and test datasets for one task."""
    get_task_spec(task_name)
    data_dir = Path(data_dir)
    if task_name == "dtd":
        return _build_dtd_datasets(data_dir)
    if task_name == "eurosat":
        return _build_eurosat_datasets(data_dir)
    return _build_dsprites_datasets(data_dir)


def build_dataloaders(
    task_name: str,
    data_dir: str | Path = DATA_DIR,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build deterministic data loaders for one Mini-VTAB task."""
    train_dataset, val_dataset, test_dataset = build_datasets(task_name, data_dir)
    pin_memory = torch.cuda.is_available()
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader
