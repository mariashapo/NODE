"""Quick helper to list and summarize result pickles."""

import argparse
import pickle
from pathlib import Path


def summarize_pickles(folder: Path):
    paths = sorted(folder.glob("**/*.pkl"))
    print(f"Scanning {folder} for pickles... found {len(paths)} file(s).")
    if not paths:
        return

    for p in paths:
        try:
            obj = pickle.load(p.open("rb"))
        except Exception as exc:
            print(f"Failed to load {p}: {exc}")
            continue

        print(f"\n=== {p} ===")
        if isinstance(obj, dict):
            printed = False
            # If this is a dict-of-dicts, print a short summary per entry
            for key, val in obj.items():
                if isinstance(val, dict):
                    printed = True
                    brief = {k: val.get(k) for k in ("mse_train", "mse_test", "time_elapsed", "termination") if k in val}
                    print(f"key={key}: {brief}")
            # Otherwise print top-level scalar metrics
            for k in ("mse_train", "mse_test", "time_elapsed", "termination", "data_type", "max_iter", "pretrain"):
                if k in obj:
                    printed = True
                    print(f"{k}: {obj[k]}")
            if not printed:
                print(f"Top-level keys: {list(obj.keys())[:10]}")
        else:
            print(f"type: {type(obj)} keys/len: {len(obj) if hasattr(obj, '__len__') else 'n/a'}")


def main():
    ap = argparse.ArgumentParser(description="Summarize result pickles in a folder.")
    ap.add_argument("folder", type=Path, help="Folder containing .pkl result files")
    args = ap.parse_args()
    summarize_pickles(args.folder)


if __name__ == "__main__":
    main()
