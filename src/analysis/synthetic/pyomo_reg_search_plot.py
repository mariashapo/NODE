"""Aggregate results from Pyomo regularization/width/tolerance sweeps.

Loads pickle files from the synthetic analysis results folder
(`src/analysis/synthetic/results/study_vdp_reg/pyomo_vdp` by default) and
produces a DataFrame that can be explored or saved to CSV.
"""

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils.analyse_results import Graphs
from math import sqrt
import numpy as np
import json
from typing import Union
import sys

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
# "study_vdp_reg/pyomo_vdp"
DEFAULT_CLI_DIR = RESULTS_ROOT / "study_ho_reg/pyomo_ho_241225"

DEFAULT_LEBEL_FONT_SIZE = 20
DEFAULT_TICK_FONT_SIZE = 16

def _resolve_results_dir(dir_path: Union[str, Path]) -> Path:
    """Resolve a user-supplied results directory, preferring synthetic/results."""
    path = Path(dir_path)
    if path.is_absolute():
        return path

    # If the caller passed "results/..." drop the prefix so we don't duplicate it
    relative = Path(*path.parts[1:]) if path.parts and path.parts[0] == "results" else path
    candidate = RESULTS_ROOT / relative
    if candidate.exists():
        return candidate
    # Fall back to the user-provided relative path if it exists; otherwise stick with the synthetic path
    return path if path.exists() else candidate


def load_reg_search(dir_path: Union[str, Path]) -> pd.DataFrame:
    """Load all .pkl result files into a flat DataFrame (one row per run).

    Handles both (layer_widths, reg, tol) keys and wall-time sweeps that use
    (data_type, pre_initialize, max_wall_time) keys. For the latter, we also
    pull defaults like layer_width/penalty_lambda_reg/tol from run_meta.json
    if present in the folder so downstream plotting still has sensible columns.
    """
    dir_path = _resolve_results_dir(dir_path)
    # Optional defaults from a companion run_meta.json
    meta_defaults = {}
    meta_path = dir_path / "run_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # promote a few handy fields if present
            for k in ("layer_width", "penalty_lambda_reg", "tol", "data_type"):
                if k in meta:
                    meta_defaults[k] = meta[k]
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read {meta_path}: {exc}")
            meta_defaults = {}

    records = []
    for fp in sorted(dir_path.glob("*.pkl")):
        try:
            run = pickle.load(open(fp, "rb"))
        except Exception as exc:  # pragma: no cover
            print(f"Failed to load {fp}: {exc}")
            continue

        for key, val in run.items():
            # Keys are (layer_widths, reg, tol); layer_widths may already be a tuple
            if not isinstance(key, (tuple, list)) or len(key) != 3:
                print(f"Skipping unexpected key {key} in {fp}")
                continue

            # Two known key layouts:
            # 1) (layer_widths, reg, tol)
            # 2) (data_type, pre_initialize, max_wall_time)
            if isinstance(key[0], str) and isinstance(key[1], (bool, np.bool_)):
                data_type, pre_init, max_wall_time = key
                lw = meta_defaults.get("layer_width")
                reg = meta_defaults.get("penalty_lambda_reg")
                tol = meta_defaults.get("tol")
            else:
                lw, reg, tol = key
                data_type = meta_defaults.get("data_type")
                pre_init = None
                max_wall_time = None

            def _get(name, default=float("nan")):
                return val.get(name, default) if isinstance(val, dict) else default

            lw_clean = tuple(lw) if isinstance(lw, (list, tuple)) else lw
            records.append(
                {
                    "file": fp.name,
                    "layer_widths": lw_clean,
                    "data_type": data_type,
                    "pre_initialize": pre_init,
                    "max_wall_time": max_wall_time,
                    "penalty_lambda_reg": reg,
                    "tol": tol,
                    "mse_train": float(_get("mse_train")),
                    "mse_test": float(_get("mse_test")),
                    "mse_train_coll": _get("mse_train_coll"),
                    "mse_test_coll": _get("mse_test_coll"),
                    "time_elapsed": _get("time_elapsed"),
                    "termination": str(_get("termination")),
                    "seed": _get("seed"),
                }
            )

    return pd.DataFrame.from_records(records)


def aggregate_by_hparams(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (layer_widths, penalty_lambda_reg, tol) and average metrics + 95% CIs."""
    if df.empty:
        return df
    # Drop rows where all key metrics are NaN so n_runs reflects valid entries
    metric_cols = ["mse_train", "mse_test", "mse_train_coll", "mse_test_coll", "time_elapsed"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    if metric_cols:
        df = df.dropna(subset=metric_cols, how="all")
    if df.empty:
        return df
    group_cols = [
        c for c in ["layer_widths", "penalty_lambda_reg", "tol", "data_type", "pre_initialize", "max_wall_time"]
        if c in df.columns and not df[c].isna().all()
    ]

    def _ci(series: pd.Series):
        n = series.count()
        if n <= 1:
            return (pd.NA, pd.NA)
        mean = series.mean()
        std = series.std(ddof=1)
        half = 1.96 * std / sqrt(n)
        return (mean - half, mean + half)

    agg_df = (
        df.groupby(group_cols, dropna=False)
          .agg(
              mse_train_mean=("mse_train", "mean"),
              mse_test_mean=("mse_test", "mean"),
              mse_train_coll_mean=("mse_train_coll", "mean"),
              mse_test_coll_mean=("mse_test_coll", "mean"),
              time_elapsed_mean=("time_elapsed", "mean"),
              mse_train_ci=("mse_train", _ci),
              mse_test_ci=("mse_test", _ci),
              mse_train_coll_ci=("mse_train_coll", _ci),
              mse_test_coll_ci=("mse_test_coll", _ci),
              time_elapsed_ci=("time_elapsed", _ci),
              n_runs=("file", "count"),
          )
          .reset_index()
    )
    # split CI tuples into separate columns for easier plotting
    def _split_ci(colname: str):
        vals = agg_df[colname]
        # Normalize to (lo, hi) tuples; if malformed, use (nan, nan)
        normed = []
        for v in vals:
            if isinstance(v, (tuple, list)) and len(v) == 2:
                lo, hi = v
                lo = np.nan if pd.isna(lo) else lo
                hi = np.nan if pd.isna(hi) else hi
                normed.append((lo, hi))
            else:
                normed.append((np.nan, np.nan))
        arr = np.array(normed, dtype=float)
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=-1) if arr.size == 2 else np.full((len(vals), 2), np.nan)
        elif arr.shape[1] != 2:
            arr = np.full((len(vals), 2), np.nan)
        agg_df[[f"{colname}_lo", f"{colname}_hi"]] = pd.DataFrame(arr, index=agg_df.index)
        agg_df.drop(columns=[colname], inplace=True)

    for col in ("mse_train_ci", "mse_test_ci", "time_elapsed_ci", "mse_train_coll_ci", "mse_test_coll_ci"):
        _split_ci(col)
    return agg_df


def _pretty_metric(name: str) -> str:
    mapping = {
        "mse_test": "MSE Test",
        "mse_train": "MSE Train",
        "mse_test_coll": "MSE Test",
        "mse_train_coll": "MSE Train",
        "time_elapsed": "Time Elapsed (s)",
    }
    return mapping.get(name, name.replace("_", " ").title())

def _format_layer_width(lw) -> str:
    """Human-readable layer width label."""
    if isinstance(lw, (list, tuple)):
        # special-case [2, w, 2] -> just the middle width
        if len(lw) == 3 and lw[0] == 2 and lw[-1] == 2:
            return str(lw[1])
        return "x".join(str(x) for x in lw)
    return str(lw)


def main(argv=None):
    import argparse
    import ast

    ap = argparse.ArgumentParser(description="Aggregate Pyomo reg/width/tol sweeps with CIs.")
    ap.add_argument(
        "--dir",
        default=str(DEFAULT_CLI_DIR),
        help="Folder with .pkl results (relative paths are resolved under src/analysis/synthetic/results).",
    )
    ap.add_argument("--plot", action="store_true", help="Plot a reg curve with CIs (pick metric/tol/layer_width).")
    ap.add_argument(
        "--metric",
        required=True,
        choices=["mse_test", "mse_train", "mse_test_coll", "mse_train_coll", "time_elapsed"],
        help="Metric to plot when --plot is set.",
    )
    ap.add_argument("--tol", type=float, default=None, help="Filter tol value (used unless --x_axis tol). Defaults to first tol present.")
    ap.add_argument("--reg", type=float, default=None, help="Filter regularization value when plotting tol/width curves.")
    ap.add_argument("--layer_width", type=str, default=None, help="Optional layer width to filter, e.g. \"[2,32,2]\".")
    ap.add_argument("--x_axis", choices=["reg", "tol", "width"], default="reg", help="Which hyperparameter to use on the x-axis.")
    ap.add_argument("--no_title", action="store_true", help="Disable title on the plot.")
    ap.add_argument("--show_points", action="store_true", help="Show point markers (default hidden to highlight error bars).")
    ap.add_argument("--boxplot", action="store_true", help="Plot per-reg boxplots for the chosen metric.")
    ap.add_argument("--inset", action="store_true", help="Add a zoomed inset for high lambda region.")
    ap.add_argument("--inset_min_x", type=float, default=1e-2, help="Lower bound for inset mask on x (log scale).")
    ap.add_argument("--inset_max_x", type=float, default=None, help="Upper bound for inset mask on x (log scale).")
    ap.add_argument("--min_runs", type=int, default=3, help="Minimum runs required per point (default: 3).")
    ap.add_argument("--label_fontsize", type=int, default=None, help="Axis label font size override.")
    ap.add_argument("--tick_fontsize", type=int, default=None, help="Tick label font size override.")
    ap.add_argument("--title_fontsize", type=int, default=None, help="Title font size override.")
    ap.add_argument("--save", action="store_true", help="Save plot to results/plots/regularization_study instead of showing.")
    ap.add_argument("--outdir", type=Path, default=Path("results/plots/regularization_study"), help="Directory to save plots when --save is used.")
    args = ap.parse_args(argv)

    df = load_reg_search(args.dir)
    if df.empty:
        print("No records loaded.")
        return

    agg = aggregate_by_hparams(df)
    print("Aggregated across seeds (mean metrics):")
    print(agg)

    if not agg.empty:
        best = agg.sort_values("mse_test_mean").head(10)
        print("\nTop 10 by mse_test_mean:")
        print(best[["layer_widths", "penalty_lambda_reg", "tol", "mse_test_mean", "mse_train_mean", "n_runs"]])

    # Shared plotting parameters (used by plot/boxplot)
    if (args.plot or args.boxplot) and not agg.empty:
        outpath = None
        if args.save:
            outdir = args.outdir
            outdir.mkdir(parents=True, exist_ok=True)
            dir_label = Path(args.dir).name
            lw_str = str(args.layer_width).replace(" ", "") if args.layer_width else "ALL"
            reg_str = f"reg{args.reg}" if args.reg is not None else "regALL"
            tol_str = f"tol{args.tol}" if args.tol is not None else "tolALL"
            kind = "boxplot" if args.boxplot else "line"
            outname = f"{dir_label}_{kind}_{args.metric}_{args.x_axis}_{reg_str}_{tol_str}_{lw_str}.png"
            outpath = outdir / outname
        target_tol = args.tol if args.tol is not None else (agg["tol"].iloc[0] if "tol" in agg.columns else None)
        target_reg = args.reg if args.reg is not None else (agg["penalty_lambda_reg"].iloc[0] if "penalty_lambda_reg" in agg.columns else None)
        lw_filter = None
        if args.layer_width is not None:
            try:
                lw_filter = tuple(ast.literal_eval(args.layer_width))
            except Exception:
                print(f"Could not parse --layer_width '{args.layer_width}', ignoring filter.")

        metric = args.metric.strip()
        metric_pretty_global = _pretty_metric(metric)
        print(f"Plotting metric: {metric_pretty_global} (raw: {metric})")

        if args.boxplot:
            if args.x_axis == "reg":
                df_box = df[df["tol"] == target_tol]
                if lw_filter is not None:
                    df_box = df_box[df_box["layer_widths"] == lw_filter]
                label_col = "penalty_lambda_reg"
                title = None if args.no_title else f"{metric_pretty_global} vs λ (tol={target_tol}, lw={lw_filter or 'ALL'})"
                x_label = "λ"
            elif args.x_axis == "tol":
                df_box = df[df["penalty_lambda_reg"] == target_reg]
                if lw_filter is not None:
                    df_box = df_box[df_box["layer_widths"] == lw_filter]
                label_col = "tol"
                title = None if args.no_title else f"{metric_pretty_global} vs tol (λ={target_reg}, lw={lw_filter or 'ALL'})"
                x_label = "tol"
            else:  # width boxplot
                df_box = df[(df["penalty_lambda_reg"] == target_reg) & (df["tol"] == target_tol)]
                label_col = "layer_widths"
                title = None if args.no_title else f"{metric_pretty_global} vs width (λ={target_reg}, tol={target_tol})"
                x_label = "Layer width"

            if df_box.empty:
                print(f"No records for boxplot with filters reg={target_reg}, tol={target_tol}, lw={lw_filter or 'ANY'}.")
                return

            data, labels = [], []
            for lbl, grp in df_box.groupby(label_col):
                vals = grp[metric].dropna().values
                if vals.size < args.min_runs:
                    continue
                data.append(vals)
                labels.append(_format_layer_width(lbl) if label_col == "layer_widths" else lbl)
            if not data:
                print(f"No data to plot boxplots after filtering (min_runs={args.min_runs}).")
                return
            Graphs.plot_single_boxplot(
                data,
                labels,
                title=title,
                ylabel=metric_pretty_global,
                x_label=x_label,
                y_log=True,
                color="C0",
                label="",
                label_fontsize=args.label_fontsize or DEFAULT_LEBEL_FONT_SIZE,
                tick_fontsize=args.tick_fontsize or DEFAULT_TICK_FONT_SIZE,
                title_fontsize=args.title_fontsize or DEFAULT_LEBEL_FONT_SIZE,
            )
            if outpath:
                plt.savefig(outpath, dpi=200, bbox_inches="tight")
                print(f"Saved plot to {outpath}")
            return

        # curve plot path
        metric_pretty = _pretty_metric(metric)
        y_col = f"{metric}_mean"
        lo_col = f"{metric}_ci_lo"
        hi_col = f"{metric}_ci_hi"
        if y_col not in agg.columns or lo_col not in agg.columns or hi_col not in agg.columns:
            print(f"Columns for metric '{metric}' not found in aggregated data.")
            return

        x_axis = args.x_axis
        yscale = "linear" if metric.startswith("time") else "log"
        if x_axis == "reg":
            sub = agg[agg["tol"] == target_tol]
            xlabel = "λ"
            xscale = "log"
            x_col = "penalty_lambda_reg"
            title_suffix = f"tol={target_tol}"
        elif x_axis == "tol":
            sub = agg[agg["penalty_lambda_reg"] == target_reg]
            xlabel = "tol"
            xscale = "log"
            x_col = "tol"
            title_suffix = f"λ={target_reg}"
        else:  # width on x-axis
            sub = agg[(agg["penalty_lambda_reg"] == target_reg) & (agg["tol"] == target_tol)]
            xlabel = "Layer width"
            xscale = "linear"
            x_col = "layer_widths"
            title_suffix = f"λ={target_reg}, tol={target_tol}"

        if lw_filter is not None and x_axis != "width":
            sub = sub[sub["layer_widths"] == lw_filter]

        if sub.empty:
            print(f"No rows for plot with filters reg={target_reg}, tol={target_tol}, layer_width={lw_filter or 'ANY'}.")
            return

        if x_axis == "width":
            g = sub[(sub["n_runs"] >= args.min_runs)].sort_values("layer_widths").copy()
            # Keep rows even if CI has NaNs; fall back to mean
            g[lo_col] = g[lo_col].fillna(g[y_col])
            g[hi_col] = g[hi_col].fillna(g[y_col])
            g = g.dropna(subset=[y_col])
            if g.empty:
                print("No valid rows to plot after filtering n_runs>=3/NA for width axis.")
                return
            labels = [_format_layer_width(lw) for lw in g["layer_widths"]]
            x_vals = np.arange(len(g))
            fig, ax = plt.subplots(figsize=(10, 6))
            Graphs.plot_reg_curve_ci(
                x_vals,
                g[y_col],
                g[lo_col],
                g[hi_col],
                title=f"{metric_pretty} vs Width ({title_suffix}; arch=[2,w,2])" if not args.no_title else None,
                xlabel="Width (architecture [2,w,2])",
                ylabel=f"{metric_pretty}",
                xscale=xscale,
                yscale=yscale,
                add_errorbars=False,
                show_points=args.show_points,
                title_on=not args.no_title,
                preserve_label_case=True,
                inset=False,
                ax=ax,
                label_fontsize=args.label_fontsize or DEFAULT_LEBEL_FONT_SIZE,
                title_fontsize=args.title_fontsize or DEFAULT_LEBEL_FONT_SIZE,
                tick_fontsize=args.tick_fontsize or DEFAULT_TICK_FONT_SIZE,
            )
            ax.set_xticks(x_vals)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            fig.tight_layout()
            if outpath:
                plt.savefig(outpath, dpi=200, bbox_inches="tight")
                print(f"Saved plot to {outpath}")
            else:
                plt.show()
            return

        # If multiple layer widths remain, plot each separately for reg/tol curves
        for lw, g in sub.groupby("layer_widths"):
            # Drop rows with NA CIs/means and low run counts
            g = g[(g["n_runs"] >= args.min_runs)].dropna(subset=[y_col, lo_col, hi_col])
            if g.empty:
                print(f"No valid rows to plot for lw={lw} (metric={metric}, x={x_axis}) after filtering n_runs>={args.min_runs}/NA.")
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            Graphs.plot_reg_curve_ci(
                g[x_col],
                g[y_col],
                g[lo_col],
                g[hi_col],
                title=f"{metric_pretty} vs {xlabel} ({title_suffix}, lw={lw})" if not args.no_title else None,
                xlabel=xlabel,
                ylabel=f"{metric_pretty}",
                xscale=xscale,
                yscale=yscale,
                add_errorbars=False,
                show_points=args.show_points,
                title_on=not args.no_title,
                preserve_label_case=True,
                inset=args.inset if x_axis == "reg" else False,
                inset_min_x=args.inset_min_x,
                inset_max_x=args.inset_max_x,
                label_fontsize=args.label_fontsize or DEFAULT_LEBEL_FONT_SIZE,
                title_fontsize=args.title_fontsize or DEFAULT_LEBEL_FONT_SIZE,
                tick_fontsize=args.tick_fontsize or DEFAULT_TICK_FONT_SIZE,
                ax=ax,
            )
            fig.tight_layout()
            if outpath:
                lw_tag = _format_layer_width(lw).replace(",", "x").replace(" ", "")
                target_path = outpath.with_name(f"{outpath.stem}_lw{lw_tag}{outpath.suffix}")
                plt.savefig(target_path, dpi=200, bbox_inches="tight")
                print(f"Saved plot to {target_path}")
                plt.close(fig)
            else:
                plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:            # invoked with CLI args
        main() 
    else:
        main([
            "--dir", str(DEFAULT_CLI_DIR),
            "--plot",
            "--metric", "mse_test_coll",
            "--boxplot"
        ])
