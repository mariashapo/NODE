"""
Pareto scatter: time_elapsed vs MSE (train/test) comparing Pyomo vs JAX per run.

Example:
  python scripts/pareto_pyomo_jax.py \
    --pyomo_dir results/study_vdp/pyomo_layer_width \
    --jax_dir results/study_vdp/jax_layer_width \
    --metric mse_test \
    --only_single_hidden \
    --recursive \
    --out results/plots/pareto_pyomo_jax.png
"""

import argparse
import pickle
from pathlib import Path
import numpy as np

from analysis.synthetic.pyomo_reg_search_plot import load_reg_search


def load_jax_folder(folder: Path, recursive: bool = False):
    vals = []
    globber = folder.rglob if recursive else folder.glob
    for fp in sorted(globber("*.pkl")):
        try:
            obj = pickle.load(fp.open("rb"))
        except Exception:
            continue
        vals.append(obj)
    return vals


def flatten_pyomo(df, metric, only_single_hidden):
    rows = []
    for _, r in df.iterrows():
        lw = r.get("layer_widths")
        if only_single_hidden and not (isinstance(lw, (list, tuple)) and len(lw) == 3):
            continue
        rows.append(
            {
                "method": "Pyomo",
                "layer_widths": lw,
                "time_elapsed": float(r.get("time_elapsed")) if r.get("time_elapsed") is not None else np.nan,
                metric: float(r.get(metric)) if r.get(metric) is not None else np.nan,
            }
        )
    return rows


def flatten_jax(objs, metric, only_single_hidden):
    rows = []

    def _scalar(val):
        if isinstance(val, (list, tuple)):
            nums = [x for x in val if isinstance(x, (int, float))]
            return float(np.sum(nums)) if nums else np.nan
        try:
            return float(val)
        except Exception:
            return np.nan
        return val

    for obj in objs:
        if isinstance(obj, dict):
            if "layer_widths" in obj:
                lw = obj.get("layer_widths")
                if only_single_hidden and not (isinstance(lw, (list, tuple)) and len(lw) == 3):
                    continue
                rows.append(
                    {
                        "method": "JAX",
                        "layer_widths": lw,
                        "time_elapsed": _scalar(obj.get("time_elapsed")),
                        metric: _scalar(obj.get(metric, np.nan)),
                    }
                )
            else:
                for k, v in obj.items():
                    lw = None
                    if isinstance(k, (list, tuple)) and len(k) >= 1:
                        lw = k[0]
                    if only_single_hidden and not (isinstance(lw, (list, tuple)) and len(lw) == 3):
                        continue
                    if isinstance(v, dict):
                        rows.append(
                            {
                                "method": "JAX",
                                "layer_widths": lw,
                                "time_elapsed": _scalar(v.get("time_elapsed")),
                                metric: _scalar(v.get(metric, np.nan)),
                            }
                        )
        # skip unsupported types
    return rows


def _width_label(lw):
    if isinstance(lw, (list, tuple)) and len(lw) == 3 and lw[0] == 2 and lw[-1] == 2:
        return str(lw[1])
    if isinstance(lw, (list, tuple)):
        return "x".join(str(x) for x in lw)
    return str(lw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pareto scatter (time vs MSE) for Pyomo vs JAX.")
    ap.add_argument("--pyomo_dir", required=True, type=Path, help="Folder with Pyomo .pkl results.")
    ap.add_argument("--jax_dir", required=True, type=Path, help="Folder with JAX .pkl results.")
    ap.add_argument("--metric", default="mse_test", help="Metric to plot on y-axis (e.g., mse_test, mse_train).")
    ap.add_argument("--only_single_hidden", action="store_true", help="Filter to [2,w,2]-style widths.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when loading pickles.")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save the figure (png).")
    ap.add_argument("--min_runs", type=int, default=1, help="Minimum runs per method/width to include.")
    ap.add_argument("--y_log", action="store_true", default=True, help="Use log scale for y (default on).")
    ap.add_argument("--x_log", action="store_true", default=False, help="Use log scale for x (time).")
    ap.add_argument("--label_widths", type=str, default=None, help="Comma-separated widths to label (e.g., '4,16,32,64,128'); if omitted, label all frontier widths.")
    ap.add_argument("--no_labels", action="store_true", help="Do not annotate frontier points with widths.")
    ap.add_argument("--label_spacing", type=float, default=0.25, help="Min time gap (s) between labels on the frontier.")
    ap.add_argument("--mse_tol", type=float, default=1e-3, help="Relative improvement required to extend the frontier (e.g., 1e-3).")
    args = ap.parse_args(argv)

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    df_pyomo = load_reg_search(str(args.pyomo_dir))
    pyomo_rows = flatten_pyomo(df_pyomo, args.metric, args.only_single_hidden)

    jax_objs = load_jax_folder(args.jax_dir, recursive=args.recursive)
    jax_rows = flatten_jax(jax_objs, args.metric, args.only_single_hidden)

    import pandas as pd

    df = pd.DataFrame(pyomo_rows + jax_rows)
    print(df)
    # ensure numeric
    df["time_elapsed"] = pd.to_numeric(df["time_elapsed"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=["time_elapsed", args.metric])
    if df.empty:
        raise SystemExit("No data to plot (time or metric missing).")
    # Normalize layer_widths to tuples for hashing/grouping
    df["layer_widths"] = df["layer_widths"].apply(lambda w: tuple(w) if isinstance(w, (list, tuple)) else w)

    # Require min_runs per method/width
    counts = df.groupby(["method", "layer_widths"]).size().reset_index(name="cnt")
    valid_pairs = counts[counts["cnt"] >= args.min_runs][["method", "layer_widths"]]
    df = df.merge(valid_pairs, on=["method", "layer_widths"], how="inner")
    # Require widths present in both methods
    widths_pyomo = set(df[df["method"] == "Pyomo"]["layer_widths"])
    widths_jax = set(df[df["method"] == "JAX"]["layer_widths"])
    common_widths = widths_pyomo & widths_jax
    df = df[df["layer_widths"].isin(common_widths)]
    if df.empty:
        raise SystemExit("No data left after filtering min_runs/common widths.")

    # Split by method
    colors = {"Pyomo": "C0", "JAX": "C1"}
    markers = {"Pyomo": "o", "JAX": "s"}

    plt.figure(figsize=(8, 6))

    # Scatter all points faintly for context
    for method, g in df.groupby("method"):
        plt.scatter(
            g["time_elapsed"],
            g[args.metric],
            label=f"{method} (all)",
            c=colors.get(method, None),
            marker=markers.get(method, "o"),
            alpha=0.2,
            s=30,
        )

    # True frontier per method: running min over all points
    min_spacing = args.label_spacing  # seconds for annotation spacing
    label_subset = None
    if args.label_widths:
        label_subset = {w.strip() for w in args.label_widths.split(",") if w.strip()}
    mse_tol = args.mse_tol  # relative improvement required to extend frontier

    frontier_ann = []
    for method, g in df.groupby("method"):
        g_sorted = g.sort_values("time_elapsed")
        best = np.inf
        frontier_rows = []
        for _, row in g_sorted.iterrows():
            val = row[args.metric]
            if val < best * (1 - mse_tol):
                best = val
                frontier_rows.append(row)
        frontier = pd.DataFrame(frontier_rows)
        if frontier.empty:
            continue
        plt.plot(
            frontier["time_elapsed"],
            frontier[args.metric],
            linestyle="-",
            marker=markers.get(method, "o"),
            color=colors.get(method, None),
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"{method} frontier",
        )
        if args.no_labels:
            continue
        # pick annotations with spacing / optional subset
        last_t = -np.inf
        last_lbl = None
        for _, row in frontier.iterrows():
            lbl = _width_label(row["layer_widths"])
            if label_subset is not None and lbl not in label_subset:
                continue
            t = row["time_elapsed"]
            if t - last_t >= min_spacing and lbl != last_lbl:
                frontier_ann.append(row)
                last_t = t
                last_lbl = lbl

    # Annotate frontier points with widths
    if frontier_ann and (not args.no_labels):
        frontier_all = pd.DataFrame(frontier_ann)
        for _, r in frontier_all.iterrows():
            plt.annotate(
                _width_label(r["layer_widths"]),
                (r["time_elapsed"], r[args.metric]),
                fontsize=20,
                alpha=0.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )

    plt.xlabel("Training Time (s)", fontsize=20)
    ylabel = args.metric.replace("_", " ").title()
    if "mse" in args.metric.lower():
        parts = []
        for p in args.metric.split("_"):
            if p.lower() == "mse":
                parts.append("MSE")
            else:
                parts.append(p.title())
        ylabel = " ".join(parts)
    plt.ylabel(ylabel, fontsize=20)
    if args.x_log:
        plt.xscale("log")
    if args.y_log:
        plt.yscale("log")
    plt.grid(True, which="major", linestyle="--", alpha=0.25)
    plt.tick_params(labelsize=18)
    plt.legend(
        frameon=False,
        fontsize=20,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),  # figure coords
        bbox_transform=plt.gcf().transFigure,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
