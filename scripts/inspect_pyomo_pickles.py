"""Lightweight inspector for Pyomo result pickles.

Prints per-file keys and a few scalar metrics to quickly spot NaNs/terminations.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable


def _format_metrics(metrics: Dict[str, Any]) -> str:
    keys = ("mse_train", "mse_test", "time_elapsed", "termination")
    parts = []
    for k in keys:
        parts.append(f"{k}={metrics.get(k)}")
    return ", ".join(parts)


def inspect_file(path: Path, limit: int) -> None:
    try:
        obj = pickle.load(path.open("rb"))
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Failed to load {path}: {exc}")
        return

    print(f"\n=== {path} ===")
    if not isinstance(obj, dict):
        print(f"type={type(obj)} (not a dict), len={len(obj) if hasattr(obj, '__len__') else 'n/a'}")
        return

    items: Iterable = list(obj.items())
    if limit > 0:
        items = items[:limit]

    for key, val in items:
        if isinstance(val, dict):
            print(f"key={key}: {_format_metrics(val)}")
        else:
            print(f"key={key}: type={type(val)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inspect Pyomo pickle files for quick metrics/terminations.")
    ap.add_argument("folder", type=Path, help="Folder containing .pkl files")
    ap.add_argument("--limit", type=int, default=5, help="Max entries per file to print (0=all)")
    args = ap.parse_args(argv)

    paths = sorted(p for p in args.folder.glob("*.pkl"))
    if not paths:
        print(f"No .pkl files found in {args.folder}")
        return

    for p in paths:
        inspect_file(p, args.limit if args.limit > 0 else 10_000_000)


if __name__ == "__main__":
    main()
