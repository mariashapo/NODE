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
            mse_test_coll_ci=("mse_train_coll", _ci),
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


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate Pyomo reg/width/tol sweeps with CIs.")
    ap.add_argument("--dir", default="results/study_vdp_reg/pyomo_vdp", help="Folder with .pkl results")
    ap.add_argument("--plot", action="store_true", help="Plot mse_test with error bars")
    args = ap.parse_args()

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

    if args.plot and not agg.empty:
        # Example: plot mse_test vs reg for each layer_width, fixed tol
        tol_vals = agg["tol"].unique()
        tol = tol_vals[0]
        sub = agg[agg["tol"] == tol]
        plt.figure(figsize=(8, 5))
        for lw, g in sub.groupby("layer_widths"):
            Graphs.plot_reg_curve_ci(
                g["penalty_lambda_reg"],
                g["mse_train_coll_mean"],
                g["mse_train_coll_ci_lo"],
                g["mse_test_coll_ci_hi"],
                title=f"mse_test vs reg (tol={tol})",
                xlabel="penalty_lambda_reg",
                ylabel="mse_test (mean ± 95% CI)",
                xscale="log",
                yscale="linear",
            )
            plt.legend([f"width {lw}"], frameon=False)
        plt.show()


if __name__ == "__main__":
    main()
