"""
Split mixed JAX result folders into width-specific subfolders by run order.

Assumes each folder contains runs for two layer widths executed sequentially
(first width, then second), with filenames containing timestamps. We split the
files in half (by modification time) and move them into `_w32` and `_w64`
subfolders. Adjust suffixes or split logic as needed.
"""

from pathlib import Path
import shutil

BASE = Path("results/study_vdp")
FOLDERS = [
    BASE / "jax_vdp_500_10000",
    BASE / "jax_vdp_1000_10000",
    BASE / "jax_vdp_15000",
]

DEST_SUFFIXES = ("w32", "w64")  # first half, second half


def split_folder(folder: Path, suffixes=DEST_SUFFIXES):
    files = sorted(folder.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"No files in {folder}, skipping.")
        return

    n = len(files)
    split = n // 2
    if split == 0:
        print(f"Not enough files in {folder} to split (n={n}).")
        return

    dest_a = folder.parent / f"{folder.name}_{suffixes[0]}"
    dest_b = folder.parent / f"{folder.name}_{suffixes[1]}"
    dest_a.mkdir(parents=True, exist_ok=True)
    dest_b.mkdir(parents=True, exist_ok=True)

    for fp in files[:split]:
        shutil.move(str(fp), dest_a / fp.name)
    for fp in files[split:]:
        shutil.move(str(fp), dest_b / fp.name)

    print(f"Split {folder} -> {dest_a} ({split} files), {dest_b} ({n - split} files)")


def main():
    for folder in FOLDERS:
        if folder.exists():
            split_folder(folder)
        else:
            print(f"{folder} does not exist, skipping.")


if __name__ == "__main__":
    main()
