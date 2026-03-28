#!/usr/bin/env python3
"""Aggregate validation metrics across runs.

1) Training logs: mean/std of "Best Val Top-1" from train_seed_*.log
2) Eval JSON: mean/std of top-1 and macro-F1 from eval_metrics.json

Usage
-----
    python scripts/aggregate_results.py runs/repro
    python scripts/aggregate_results.py runs/repro --eval-json
"""
import json
import re
import statistics
import sys
from pathlib import Path

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


def find_eval_json_top1(folder: Path):
    """Collect top-1 and macro-F1 from logs/**/eval_metrics.json."""
    top1, f1m = [], []
    for p in sorted(folder.glob("**/eval_metrics.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        c = data.get("classification") or {}
        if "top1_accuracy_pct" in c:
            top1.append(float(c["top1_accuracy_pct"]))
        if "f1_macro" in c:
            f1m.append(float(c["f1_macro"]))
    return top1, f1m


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate_results.py <runs_folder> [--eval-json]")
        sys.exit(1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    if not args:
        print("Usage: python scripts/aggregate_results.py <runs_folder> [--eval-json]")
        sys.exit(1)
    folder = Path(args[0])
    if not folder.exists():
        print("Folder not found:", folder)
        sys.exit(2)

    if "--eval-json" in flags:
        top1, f1m = find_eval_json_top1(folder)
        if not top1:
            print("No eval_metrics.json with top1 found under", folder)
            sys.exit(3)
        print("Top-1 (%):", top1)
        m1 = statistics.mean(top1)
        s1 = statistics.pstdev(top1) if len(top1) > 1 else 0.0
        print(f"Top-1 mean ± pop-std: {m1:.4f} ± {s1:.4f}")
        if f1m:
            print("Macro-F1:", f1m)
            mf = statistics.mean(f1m)
            sf = statistics.pstdev(f1m) if len(f1m) > 1 else 0.0
            print(f"Macro-F1 mean ± pop-std: {mf:.4f} ± {sf:.4f}")
        return

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
