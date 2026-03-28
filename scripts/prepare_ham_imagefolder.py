#!/usr/bin/env python3
"""
Build ImageFolder layout for HAM10000 from the raw Dataverse download:

  <raw_dir>/*.jpg (flat or nested) + HAM10000_metadata.*

Produces::

  <output_dir>/train/<dx>/*.jpg
  <output_dir>/val/<dx>/*.jpg

Split is **lesion-level** (all images of one lesion stay in train or val) with
stratification on diagnosis ``dx``, so you avoid easy leakage from duplicates.

Usage
-----
  python scripts/prepare_ham_imagefolder.py \\
      --raw_dir data/data/ham10000_raw \\
      --output_dir data/data/ham10000

Then train with default HAM path (see config.HAM10000_DIR) or::

  python train.py --dataset ham10000 --data_root data/data/ham10000
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _detect_delimiter(sample: str) -> str:
    if sample.count("\t") > sample.count(","):
        return "\t"
    return ","


def _load_metadata_rows(meta_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    text = meta_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"Empty metadata file: {meta_path}")
    delim = _detect_delimiter(lines[0][:500])
    reader = csv.DictReader(lines, delimiter=delim)
    fieldnames = reader.fieldnames or []
    rows = []
    for row in reader:
        if not row:
            continue
        rows.append({k: (v or "").strip() for k, v in row.items()})
    if not rows:
        raise SystemExit(f"No rows parsed in {meta_path}")
    return rows, list(fieldnames)


def _column_lookup(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical names to actual header."""
    key_to_actual = {_norm_key(f): f for f in fieldnames}
    out = {}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            nk = _norm_key(c)
            if nk in key_to_actual:
                return key_to_actual[nk]
        for canon, actual in key_to_actual.items():
            for c in candidates:
                if _norm_key(c) in canon:
                    return actual
        return None

    out["lesion_id"] = pick("lesion_id", "lesionid")
    out["image_id"] = pick("image", "image_id", "imageid")
    out["dx"] = pick("dx", "diagnosis", "label")
    return out


def _image_basename(cell: str) -> str:
    cell = cell.strip()
    if not cell:
        return ""
    base = os.path.basename(cell)
    if "." not in base:
        base = f"{base}.jpg"
    return base


def _index_images(raw_dir: Path) -> dict[str, Path]:
    """basename lower -> first path found."""
    index: dict[str, Path] = {}
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
        ):
            key = p.name.lower()
            index.setdefault(key, p)
    return index


def prepare(
    raw_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    use_copy: bool,
) -> None:
    if not raw_dir.is_dir():
        raise SystemExit(f"raw_dir not found: {raw_dir}")

    meta_candidates = sorted(
        list(raw_dir.rglob("HAM10000_metadata*"))
        + list(raw_dir.rglob("*ham10000*metadata*"))
        + list(raw_dir.glob("*.csv"))
    )
    meta_path = next(
        (p for p in meta_candidates if p.suffix.lower() in (".csv", ".tab", ".tsv", ".txt")),
        None,
    )
    if meta_path is None:
        raise SystemExit(
            f"No metadata file found under {raw_dir}. "
            "Expected something like HAM10000_metadata.csv or .tab"
        )

    rows, fieldnames = _load_metadata_rows(meta_path)
    cols = _column_lookup(fieldnames)
    if not cols.get("image_id") or not cols.get("dx"):
        raise SystemExit(
            f"Could not find image + dx columns in {meta_path}. "
            f"Headers: {fieldnames}"
        )

    img_index = _index_images(raw_dir)
    if not img_index:
        raise SystemExit(f"No .jpg/.png images found under {raw_dir}")

    lesion_to_dx: dict[str, str] = {}
    lesion_to_basenames: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        dx = row.get(cols["dx"], "").strip()
        if not dx:
            continue
        img_cell = row.get(cols["image_id"], "").strip()
        base = _image_basename(img_cell)
        if not base:
            continue
        if base.lower() not in img_index:
            if base.lower().replace(".jpg", ".jpeg") in img_index:
                base = base[:-4] + ".jpeg"
            elif not any(
                k.startswith(base.lower().rsplit(".", 1)[0]) for k in img_index
            ):
                continue

        lid_col = cols.get("lesion_id")
        if lid_col and row.get(lid_col, "").strip():
            lid = row[lid_col].strip()
        else:
            lid = base

        if lid in lesion_to_dx and lesion_to_dx[lid] != dx:
            raise SystemExit(
                f"lesion_id {lid} has conflicting dx: {lesion_to_dx[lid]} vs {dx}"
            )
        lesion_to_dx[lid] = dx
        if base not in lesion_to_basenames[lid]:
            lesion_to_basenames[lid].append(base)

    if not lesion_to_dx:
        raise SystemExit("No rows matched images on disk; check image_id vs filenames.")

    lesions = list(lesion_to_dx.keys())
    y = [lesion_to_dx[l] for l in lesions]
    try:
        train_l, val_l = train_test_split(
            lesions,
            test_size=val_ratio,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        train_l, val_l = train_test_split(
            lesions, test_size=val_ratio, random_state=seed
        )

    train_set, val_set = set(train_l), set(val_l)

    for split in ("train", "val"):
        p = output_dir / split
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    missing: list[str] = []

    def link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if use_copy:
            shutil.copy2(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)

    for lid, dx in lesion_to_dx.items():
        split = "train" if lid in train_set else "val"
        for base in lesion_to_basenames[lid]:
            key = base.lower()
            src = img_index.get(key)
            if src is None:
                stem = Path(base).stem.lower()
                src = next((p for k, p in img_index.items() if k.startswith(stem)), None)
            if src is None:
                missing.append(base)
                continue
            dst = output_dir / split / dx / src.name
            link_or_copy(src, dst)
            stats[f"{split}_ok"] += 1

    print(f"Metadata   : {meta_path}")
    print(f"Images idx : {len(img_index):,} files under {raw_dir}")
    print(f"Lesions    : {len(lesion_to_dx):,} (train {len(train_set)}, val {len(val_set)})")
    print(f"Linked/copied image copies: {stats['train_ok']} train, {stats['val_ok']} val")
    if missing:
        print(f"WARNING: {len(missing)} metadata entries had no file match (showing up to 5):")
        for m in missing[:5]:
            print(f"   {m}")
    print(f"Output     : {output_dir.resolve()}")
    print("Next: python train.py --dataset ham10000 --data_root", output_dir)


def main():
    ap = argparse.ArgumentParser(description="HAM10000 → ImageFolder train/val")
    ap.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/data/ham10000_raw"),
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/data/ham10000"),
    )
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlink (slower, more disk; use on Windows if needed).",
    )
    args = ap.parse_args()
    random.seed(args.seed)
    prepare(
        args.raw_dir.resolve(),
        args.output_dir.resolve(),
        args.val_ratio,
        args.seed,
        args.copy,
    )


if __name__ == "__main__":
    main()
