#!/usr/bin/env python3
"""Simple aggregator: parse textual logs for best validation top-1 and compute mean/std.

Usage: python scripts/aggregate_results.py runs/repro

It looks for files matching train_seed_*.log inside the provided folder.
"""
import sys
import re
from pathlib import Path
import statistics

PATTERN = re.compile(r"Best Val Top-1\s*[:=]\s*([0-9]+\.?[0-9]*)")


def find_scores(logdir: Path):
    scores = []
    for p in sorted(logdir.glob("**/train_seed_*.log")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = PATTERN.search(text)
        if m:
            scores.append(float(m.group(1)))
        else:
            # try alternate patterns
            m2 = re.search(r"Val Acc[^0-9]*([0-9]+\.?[0-9]*)", text)
            if m2:
                scores.append(float(m2.group(1)))
    return scores


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_results.py <runs_folder>")
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.exists():
        print("Folder not found:", folder)
        sys.exit(2)
    scores = find_scores(folder)
    if not scores:
        print("No scores found in", folder)
        sys.exit(3)
    print("Found scores:", scores)
    mean = statistics.mean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    print(f"Mean: {mean:.4f}, Std (population): {stdev:.4f}")


if __name__ == "__main__":
    main()
