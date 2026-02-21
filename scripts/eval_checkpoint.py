#!/usr/bin/env python3
"""Evaluate a saved checkpoint on CIFAR-100 test or CIFAR-100-C (corruptions).

Usage examples:
  # CIFAR-100 test (default)
  python scripts/eval_checkpoint.py --ckpt checkpoints/best_model.pth

  # ImageFolder validation directory
  python scripts/eval_checkpoint.py --ckpt checkpoints/best_model.pth --val_data_dir /path/to/val_folder

  # CIFAR-100-C folder with .npy corruption files + labels.npy
  python scripts/eval_checkpoint.py --ckpt checkpoints/best_model.pth --c100c_dir /path/to/CIFAR-100-C

The script prints per-corruption and overall accuracies.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import config as cfg
from models.vit_freqmerge import build_freqmerge_vit
from data.cifar100 import get_val_transform
from data.val_loader import get_val_loader_from_dir
from utils.metrics import accuracy


class NumpyImageDataset(Dataset):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, transform=None):
        # imgs: (N, H, W, C) uint8 or float
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        # Convert to PIL Image
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


def load_cifar100c_loaders(c100c_dir: str, batch_size: int):
    p = Path(c100c_dir)
    if not p.exists():
        raise FileNotFoundError(f"CIFAR-100-C folder not found: {c100c_dir}")

    # Expect files like labels.npy and <corruption>.npy
    labels_path = p / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy not found in {c100c_dir}")
    labels = np.load(str(labels_path))

    transform = get_val_transform(cfg.IMAGE_SIZE)

    loaders = {}
    for npy in sorted(p.glob("*.npy")):
        name = npy.stem
        if name == "labels":
            continue
        arr = np.load(str(npy))  # expected shape (N, H, W, C)
        ds = NumpyImageDataset(arr, labels, transform=transform)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    return loaders


def evaluate_model(model, loader, device):
    model.eval()
    top1_sum = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            acc1, _ = accuracy(logits.float(), labels, topk=(1, 5))
            b = images.size(0)
            top1_sum += acc1 * b
            n += b
    return float(top1_sum / n)


def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    # Build model matching saved args where possible
    merge_layers = args.get("merge_layers", cfg.MERGE_LAYERS)
    keep_rate = args.get("keep_rate", cfg.KEEP_RATE)
    alpha = args.get("alpha", cfg.ALPHA)
    hpf_radius = args.get("hpf_radius", cfg.HPF_RADIUS)
    pretrained = args.get("pretrained", cfg.PRETRAINED)
    backbone = args.get("backbone", cfg.BACKBONE)

    model = build_freqmerge_vit(
        num_classes=cfg.NUM_CLASSES,
        merge_layers=merge_layers,
        keep_rate=keep_rate,
        alpha=alpha,
        hpf_radius=hpf_radius,
        pretrained=pretrained,
        backbone=backbone,
    ).to(device)

    # Load state dict
    state = ckpt.get("state_dict", ckpt)
    # If the state dict keys have 'module.' prefix, strip it
    if any(k.startswith("module.") for k in list(state.keys())):
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        state = new_state
    model.load_state_dict(state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_data_dir", default=None,
                   help="ImageFolder-style validation directory")
    p.add_argument("--c100c_dir", default=None, help="Path to CIFAR-100-C .npy files")
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model_from_ckpt(args.ckpt, device)

    if args.c100c_dir:
        loaders = load_cifar100c_loaders(args.c100c_dir, args.batch_size)
        results = {}
        for name, loader in loaders.items():
            acc = evaluate_model(model, loader, device)
            print(f"Corruption: {name:20s}  Top1: {acc:.2f}%")
            results[name] = acc
        mean = sum(results.values()) / len(results)
        print(f"\nMean corruption Top1: {mean:.2f}% over {len(results)} corruptions")
        return

    # Otherwise use CIFAR-100 test or ImageFolder
    if args.val_data_dir:
        loader = get_val_loader_from_dir(args.val_data_dir, batch_size=args.batch_size)
    else:
        # fallback to CIFAR-100 test split
        from data.cifar100 import get_cifar100_loaders
        _, loader = get_cifar100_loaders(batch_size=args.batch_size)

    acc = evaluate_model(model, loader, device)
    print(f"Validation Top1: {acc:.2f}%")


if __name__ == "__main__":
    main()
