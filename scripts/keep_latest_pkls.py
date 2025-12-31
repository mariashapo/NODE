"""Keep only the newest N .pkl files in a folder (dry-run by default).

Usage:
  python scripts/keep_latest_pkls.py results/study_vdp/pyomo_vdp_32_301225 --keep 15 --apply
"""

import argparse
from pathlib import Path


def collect_pkls(folder: Path):
    return sorted(folder.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)


def main():
    ap = argparse.ArgumentParser(description="Keep only the newest N .pkl files (dry-run by default).")
    ap.add_argument("folder", type=Path, help="Folder containing .pkl files.")
    ap.add_argument("--keep", type=int, default=15, help="Number of newest .pkl files to keep.")
    ap.add_argument("--apply", action="store_true", help="Actually delete older files. Omit to just print.")
    args = ap.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found: {args.folder}")

    files = collect_pkls(args.folder)
    if not files:
        print(f"No .pkl files found in {args.folder}")
        return

    keep = files[: args.keep]
    delete = files[args.keep :]

    print(f"Keeping {len(keep)} newest .pkl files (showing names):")
    for p in keep:
        print(f"  KEEP  {p.name}")

    if delete:
        print(f"\nWould delete {len(delete)} older .pkl files:")
        for p in delete:
            print(f"  DEL   {p.name}")
    else:
        print("\nNothing to delete (folder has <= keep count).")

    if args.apply and delete:
        for p in delete:
            try:
                p.unlink()
            except Exception as exc:  # pragma: no cover
                print(f"Failed to delete {p}: {exc}")
        print("\nDeletion complete.")
    elif not args.apply and delete:
        print("\nDry run only. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()
