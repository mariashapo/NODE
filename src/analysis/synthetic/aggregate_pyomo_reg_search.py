"""Aggregate results from Pyomo regularization/width/tolerance sweeps.

Loads pickle files from results/study_vdp_reg/pyomo_vdp and produces a
DataFrame that can be explored or saved to CSV.
"""

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils.analyse_results import Graphs
from math import sqrt

def load_reg_search(dir_path: str = "results/study_vdp_reg/pyomo_vdp") -> pd.DataFrame:
    """Load all .pkl result files into a flat DataFrame (one row per run)."""
    records = []
    for fp in sorted(Path(dir_path).glob("*.pkl")):
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
            lw, reg, tol = key

            def _get(name, default=float("nan")):
                return val.get(name, default) if isinstance(val, dict) else default

            records.append(
                {
                    "file": fp.name,
                    "layer_widths": lw,
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
    group_cols = ["layer_widths", "penalty_lambda_reg", "tol"]

    def _ci(series: pd.Series):
        n = series.count()
        if n <= 1:
            return (pd.NA, pd.NA)
        mean = series.mean()
        std = series.std(ddof=1)
        half = 1.96 * std / sqrt(n)
        return (mean - half, mean + half)

    agg_df = (
        df.groupby(group_cols)
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
    for col in ("mse_train_ci", "mse_test_ci", "time_elapsed_ci", "mse_train_coll_ci", "mse_test_coll_ci"):
        agg_df[[f"{col}_lo", f"{col}_hi"]] = pd.DataFrame(agg_df[col].tolist(), index=agg_df.index)
        agg_df.drop(columns=[col], inplace=True)
    return agg_df


def _pretty_metric(name: str) -> str:
    mapping = {
        "mse_test": "MSE Test",
        "mse_train": "MSE Train",
        "mse_test_coll": "MSE Test",
        "mse_train_coll": "MSE Train",
    }
    return mapping.get(name, name.replace("_", " ").title())


def main(argv=None):
    import argparse
    import ast

    ap = argparse.ArgumentParser(description="Aggregate Pyomo reg/width/tol sweeps with CIs.")
    ap.add_argument("--dir", default="results/study_ho_reg/pyomo_ho_241225", help="Folder with .pkl results")
    ap.add_argument("--plot", action="store_true", help="Plot a reg curve with CIs (pick metric/tol/layer_width).")
    ap.add_argument("--metric", required=True, choices=["mse_test", "mse_train", "mse_test_coll", "mse_train_coll"], help="Metric to plot when --plot is set.")
    ap.add_argument("--tol", type=float, default=None, help="Filter tol value to plot; defaults to the first tol present.")
    ap.add_argument("--layer_width", type=str, default=None, help="Optional layer width to filter, e.g. \"[2,32,2]\".")
    ap.add_argument("--no_title", action="store_true", help="Disable title on the plot.")
    ap.add_argument("--show_points", action="store_true", help="Show point markers (default hidden to highlight error bars).")
    ap.add_argument("--boxplot", action="store_true", help="Plot per-reg boxplots for the chosen metric.")
    ap.add_argument("--inset", action="store_true", help="Add a zoomed inset for high lambda region.")
    ap.add_argument("--inset_min_x", type=float, default=1e-2, help="Lower bound for inset mask on x (log scale).")
    ap.add_argument("--inset_max_x", type=float, default=None, help="Upper bound for inset mask on x (log scale).")
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
        target_tol = args.tol if args.tol is not None else agg["tol"].iloc[0]
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
            df_box = df[df["tol"] == target_tol]
            if lw_filter is not None:
                df_box = df_box[df_box["layer_widths"] == lw_filter]
            if df_box.empty:
                print(f"No records for boxplot at tol={target_tol} and layer_width={lw_filter or 'ANY'}.")
                return
            data = []
            labels = []
            for reg, grp in df_box.groupby("penalty_lambda_reg"):
                vals = grp[metric].dropna().values
                if vals.size == 0:
                    continue
                data.append(vals)
                labels.append(reg)
            if not data:
                print("No data to plot boxplots after filtering.")
                return
            Graphs.plot_single_boxplot(
                data,
                labels,
                title=None if args.no_title else f"{metric_pretty_global} vs λ (tol={target_tol}, lw={lw_filter or 'ALL'})",
                ylabel=metric_pretty_global,
                x_label="λ",
                y_log=True,
                color="C0",
                label="",
            )
            return

        # curve plot path
        sub = agg[agg["tol"] == target_tol]
        if lw_filter is not None:
            sub = sub[sub["layer_widths"] == lw_filter]
        if sub.empty:
            print(f"No rows for tol={target_tol} and layer_width={lw_filter or 'ANY'}.")
            return

        y_col = f"{metric}_mean"
        lo_col = f"{metric}_ci_lo"
        hi_col = f"{metric}_ci_hi"
        if y_col not in sub.columns or lo_col not in sub.columns or hi_col not in sub.columns:
            print(f"Columns for metric '{metric}' not found in aggregated data.")
            return

        # If multiple layer widths remain, plot each separately
        for lw, g in sub.groupby("layer_widths"):
            # Drop rows with NA CIs/means and low run counts
            g = g[(g["n_runs"] >= 3)].dropna(subset=[y_col, lo_col, hi_col])
            if g.empty:
                print(f"No valid rows to plot for lw={lw} (metric={metric}) after filtering n_runs>=3/NA.")
                continue
            metric_pretty = _pretty_metric(metric)
            Graphs.plot_reg_curve_ci(
                g["penalty_lambda_reg"],
                g[y_col],
                g[lo_col],
                g[hi_col],
                title=f"{metric_pretty} vs Reg (tol={target_tol}, lw={lw})" if not args.no_title else None,
                xlabel="λ",
                ylabel=f"{metric_pretty} (Mean ± 95% CI)",
                xscale="log",
                yscale="log",
                add_errorbars=False,
                show_points=args.show_points,
                title_on=not args.no_title,
                preserve_label_case=True,
                inset=args.inset,
                inset_min_x=args.inset_min_x,
                inset_max_x=args.inset_max_x,
            )


if __name__ == "__main__":
    action = "prod"
    if action == "dev":
        main([
            "--dir", "results/study_ho_reg/pyomo_ho_241225",
            "--plot",
            "--metric", "mse_test_coll",
            "--tol", "1e-6",
            "--layer_width", "[2,32,2]",
        ])
    else:    
        main()
