"""
data/cifar100.py
----------------
CIFAR-100 DataLoaders.

Key preprocessing step
----------------------
CIFAR-100 images are natively 32×32 pixels, which yields only 4 patches per
side (4×4 = 16 tokens) when tokenised with a 16×16 patch size — far too few
for meaningful token merging.  We therefore upscale every image to 224×224
via bilinear interpolation (matching the ImageNet-pretrained ViT resolution)
to obtain 14×14 = 196 spatial tokens per image.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """
    Training augmentation pipeline.

    Stages
    ------
    1. Resize to image_size × image_size  (bilinear upscale from 32 px)
    2. Random crop with padding           (mild spatial jitter)
    3. Random horizontal flip             (50 % probability)
    4. ToTensor + ImageNet normalisation  (using CIFAR-100 channel statistics
                                           which are close to ImageNet values)
    """
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomCrop(
            image_size,
            padding=cfg.RANDOM_CROP_PADDING,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
    ])


def get_val_transform(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """
    Validation / test transform pipeline (deterministic).

    No random augmentation; only resize + normalise.
    """
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_cifar100_loaders(
    data_dir:   str   = cfg.DATA_DIR,
    batch_size: int   = cfg.BATCH_SIZE,
    num_workers:int   = cfg.NUM_WORKERS,
    pin_memory: bool  = cfg.PIN_MEMORY,
    image_size: int   = cfg.IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader) for CIFAR-100.

    Parameters
    ----------
    data_dir    : Directory where CIFAR-100 will be downloaded / cached.
    batch_size  : Mini-batch size.
    num_workers : Number of parallel data-loading workers.
    pin_memory  : Pin memory for faster GPU transfers.
    image_size  : Target image resolution (both H and W).

    Returns
    -------
    (train_loader, val_loader) : Both are standard PyTorch DataLoaders.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transform(image_size),
    )
    val_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=get_val_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # avoids batch-norm issues on partial batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Data] CIFAR-100 loaded from '{data_dir}'")
    print(f"       Train samples : {len(train_dataset):,}")
    print(f"       Val   samples : {len(val_dataset):,}")
    print(f"       Image size    : {image_size}×{image_size} (upscaled from 32×32)")
    print(f"       Batch size    : {batch_size}")

    return train_loader, val_loader
