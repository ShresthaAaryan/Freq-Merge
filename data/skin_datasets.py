"""
data/skin_datasets.py
---------------------
HAM10000 and ISIC 2019 DataLoaders (ImageFolder layout).

Expected layout (each dataset root is separate):
    <root>/train/<class_name>/*.jpg
    <root>/val/<class_name>/*.jpg
If ``val`` is missing, ``test`` is used as the validation split.

HAM10000 typically has 7 lesion categories; ISIC 2019 class count follows your
folder structure. Normalisation uses ImageNet statistics (matches ViT pretraining).
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


def get_train_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(image_size, padding=cfg.RANDOM_CROP_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
        ]
    )


def get_val_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
        ]
    )


def _normalize_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("ham10000", "ham1000"):
        return "ham10000"
    if key == "isic2019":
        return "isic2019"
    raise ValueError(
        f"Unknown dataset '{name}'. Expected 'ham10000' or 'isic2019'."
    )


def _root_for_dataset(dataset: str) -> str:
    dataset = _normalize_dataset_name(dataset)
    if dataset == "ham10000":
        return cfg.HAM10000_DIR
    return cfg.ISIC2019_DIR


def get_skin_loaders(
    dataset: str = cfg.DEFAULT_SKIN_DATASET,
    data_root: str | None = None,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
    pin_memory: bool = cfg.PIN_MEMORY,
    image_size: int = cfg.IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    """
    Build train/val loaders for HAM10000 or ISIC 2019.

    Returns
    -------
    train_loader, val_loader, num_classes, class_names
    """
    root = data_root if data_root is not None else _root_for_dataset(dataset)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(val_dir):
        alt = os.path.join(root, "test")
        if os.path.isdir(alt):
            val_dir = alt

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training images not found at '{train_dir}'. "
            "Expected layout: <root>/train/<class_name>/*.jpg (and val/ or test/). "
            "Use --data_root /path/to/dataset, or set HAM10000_DIR / ISIC2019_DIR in config.py."
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation images not found at '{root}/val' or '{root}/test'."
        )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_train_transform(image_size),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_val_transform(image_size),
    )

    num_classes = len(train_dataset.classes)
    if len(val_dataset.classes) != num_classes:
        raise ValueError(
            f"Train has {len(train_dataset.classes)} classes but val has "
            f"{len(val_dataset.classes)} — folder sets must match."
        )
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and val class folder names differ. Use the same subfolder "
            "names under train/ and val/."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    label = _normalize_dataset_name(dataset)
    print(f"[Data] {label.upper()}  root='{root}'")
    print(f"       Train samples : {len(train_dataset):,}")
    print(f"       Val   samples : {len(val_dataset):,}")
    print(f"       Classes ({num_classes}) : {train_dataset.classes}")
    print(f"       Image size    : {image_size}×{image_size}")
    print(f"       Batch size    : {batch_size}")

    return train_loader, val_loader, num_classes, train_dataset.classes
