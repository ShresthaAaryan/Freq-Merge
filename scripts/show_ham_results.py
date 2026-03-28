#!/usr/bin/env python3
"""
Build a text/Markdown table from multi-seed HAM / ISIC training outputs.

Looks for:
  <run_root>/multi_run_summary.json
or scans:
  <run_root>/seed_*/logs/training_history.json

Optional: merge eval metrics if <run_root>/seed_*/logs/eval_metrics.json exists
(evaluate with --save_metrics pointing under each seed's logs/).

Usage
-----
    python scripts/show_ham_results.py runs/ham10000_seeds
    python scripts/show_ham_results.py runs/ham10000_seeds --markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def collect_from_summary(run_root: Path) -> tuple[list[dict], dict | None]:
    summ = run_root / "multi_run_summary.json"
    if summ.is_file():
        data = _load_json(summ)
        runs = data.get("runs") or []
        meta = {k: v for k, v in data.items() if k != "runs"}
        return runs, meta
    return [], None


def collect_from_histories(run_root: Path) -> list[dict]:
    rows = []
    for hist in sorted(run_root.glob("seed_*/logs/training_history.json")):
        seed_part = hist.parents[1].name  # seed_42
        try:
            seed = int(seed_part.split("_", 1)[1])
        except (IndexError, ValueError):
            seed = -1
        h = _load_json(hist)
        val = h.get("val_acc1") or []
        best = max(val) if val else float("nan")
        ep = val.index(max(val)) + 1 if val else 0
        ckpt = hist.parent.parent / "checkpoints" / "best_model.pth"
        rows.append(
            {
                "seed": seed,
                "best_val_top1": best,
                "best_epoch": ep,
                "checkpoint": str(ckpt),
                "history_path": str(hist),
                "log_dir": str(hist.parent),
            }
        )
    return rows


def try_eval_metrics(log_dir: Path) -> dict | None:
    for name in ("eval_metrics.json",):
        p = log_dir / name
        if p.is_file():
            d = _load_json(p)
            c = d.get("classification") or {}
            return {
                "top1": c.get("top1_accuracy_pct"),
                "macro_f1": c.get("f1_macro"),
                "balanced_acc": c.get("balanced_accuracy"),
            }
    return None


def main():
    ap = argparse.ArgumentParser(description="Tabulate multi-seed HAM/ISIC runs.")
    ap.add_argument("run_root", type=str, help="Folder containing seed_* or multi_run_summary.json")
    ap.add_argument("--markdown", action="store_true", help="Print GitHub-flavored Markdown table.")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_dir():
        print("Not a directory:", root, file=sys.stderr)
        sys.exit(2)

    runs, meta = collect_from_summary(root)
    if not runs:
        runs = collect_from_histories(root)
    if not runs:
        print("No multi_run_summary.json and no seed_*/logs/training_history.json under", root)
        sys.exit(3)

    runs = sorted(runs, key=lambda r: r.get("seed", 0))

    extras = []
    for r in runs:
        ld = Path(r.get("log_dir", ""))
        if ld.is_dir():
            extras.append(try_eval_metrics(ld))
        else:
            extras.append(None)

    if args.markdown:
        hdr = "| Seed | Best Val Top-1 (%) | Best epoch | Top-1 eval* | Macro-F1* |"
        sep = "|:-----|------------------:|-----------:|------------:|----------:|"
        print(hdr)
        print(sep)
        for r, ev in zip(runs, extras):
            te = ev or {}
            e1 = te.get("top1")
            f1 = te.get("macro_f1")
            print(
                f"| {r.get('seed', '')} | {r.get('best_val_top1', 0):.2f} | "
                f"{r.get('best_epoch', '')} | "
                f"{e1 if e1 is not None else '—'} | "
                f"{f1 if f1 is not None else '—'} |"
            )
        print("\n*Eval columns filled if `logs/eval_metrics.json` exists per seed.")
    else:
        print(f"Run root: {root.resolve()}")
        if meta and meta.get("best_val_top1_mean") is not None:
            m = meta["best_val_top1_mean"]
            s = meta.get("best_val_top1_std") or 0.0
            print(
                f"Dataset: {meta.get('dataset', '?')}  |  mean ± std: {m:.2f} ± {s:.2f}"
            )
        print(f"{'Seed':>6}  {'Val Top-1':>10}  {'Ep':>4}  {'eval Top-1':>12}  {'macro-F1':>10}")
        for r, ev in zip(runs, extras):
            te = ev or {}
            e1 = te.get("top1")
            f1 = te.get("macro_f1")
            e1s = f"{e1:.2f}" if e1 is not None else "—"
            f1s = f"{f1:.4f}" if f1 is not None else "—"
            print(
                f"{r.get('seed', -1):6d}  {r.get('best_val_top1', 0):10.2f}  "
                f"{r.get('best_epoch', 0):4d}  {e1s:>12}  {f1s:>10}"
            )

    print("\nCheckpoints:")
    for r in runs:
        print(f"  seed {r.get('seed')}: {r.get('checkpoint', '')}")


if __name__ == "__main__":
    main()
