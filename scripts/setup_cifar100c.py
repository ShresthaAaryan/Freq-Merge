#!/usr/bin/env python3
"""Download and prepare CIFAR-100-C corruptions as .npy files.

The script tries a small list of known mirrors; if download fails, you can
pass `--archive /path/to/CIFAR-100-C.tar` to use a local copy.

Outputs into `data/cifar100_c/` by default with files like `gaussian_noise.npy`
and `labels.npy` ready for `scripts/eval_checkpoint.py`.
"""
import argparse
import os
import tarfile
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve


KNOWN_URLS = [
    # Zenodo mirrors — try common variants; some may 404 depending on mirror
    "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar",
    "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar.gz",
    "https://people.cs.uchicago.edu/~pavlakos/data/CIFAR-100-C.tar",
]


def download_archive(url: str, target: Path):
    print(f"Downloading {url} -> {target} ...")
    urlretrieve(url, str(target))


def extract_archive(archive: Path, extract_to: Path):
    print(f"Extracting {archive} -> {extract_to}")
    with tarfile.open(str(archive), "r:*") as tf:
        tf.extractall(path=str(extract_to))


def collect_npy_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    found = False
    for p in src_dir.rglob("*.npy"):
        found = True
        target = dst_dir / p.name
        print(f"Copying {p} -> {target}")
        shutil.copyfile(str(p), str(target))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=os.path.join("data", "cifar100_c"))
    p.add_argument("--archive", default=None,
                   help="Optional local CIFAR-100-C tar archive to use instead of downloading")
    p.add_argument("--try-urls", action="store_true",
                   help="Try known URLs for automatic download (may fail behind firewalls)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    tmpdir = outdir.parent / (outdir.name + "_tmp")
    archive_path = None

    if args.archive:
        archive_path = Path(args.archive)
        if not archive_path.exists():
            print(f"Provided archive not found: {archive_path}")
            sys.exit(2)

    if archive_path is None and args.try_urls:
        tmpdir.mkdir(parents=True, exist_ok=True)
        for url in KNOWN_URLS:
            try:
                archive_path = tmpdir / Path(url).name
                download_archive(url, archive_path)
                break
            except Exception as e:
                print(f"Download failed for {url}: {e}")
                archive_path = None

    if archive_path is None:
        print("No archive available. Provide --archive or run with --try-urls to attempt download.")
        print("Example: python scripts/setup_cifar100c.py --try-urls")
        sys.exit(1)

    # Extract and collect
    try:
        extract_dir = tmpdir if tmpdir.exists() else outdir.parent / (outdir.name + "_extract")
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, extract_dir)

        # Look for .npy files under extracted folder
        found = collect_npy_files(extract_dir, outdir)
        if not found:
            print("No .npy files found in archive — archive structure may differ.")
            print(f"Inspect {extract_dir} to locate files manually.")
            sys.exit(3)

        print(f"CIFAR-100-C prepared at {outdir}")
    finally:
        # Clean up temporary extraction if created under tmpdir
        if tmpdir.exists():
            try:
                shutil.rmtree(str(tmpdir))
            except Exception:
                pass


if __name__ == "__main__":
    main()
