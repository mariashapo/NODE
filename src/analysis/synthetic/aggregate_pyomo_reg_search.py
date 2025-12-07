"""Aggregate results from Pyomo regularization/width/tolerance sweeps.

Loads pickle files from results/study_vdp_reg/pyomo_vdp and produces a
DataFrame that can be explored or saved to CSV.
"""

from pathlib import Path
import pickle
import pandas as pd


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
    """Group by (layer_widths, penalty_lambda_reg, tol) and average metrics."""
    if df.empty:
        return df
    group_cols = ["layer_widths", "penalty_lambda_reg", "tol"]
    agg_df = (
        df.groupby(group_cols)
        .agg(
            mse_train_mean=("mse_train", "mean"),
            mse_test_mean=("mse_test", "mean"),
            mse_train_coll_mean=("mse_train_coll", "mean"),
            mse_test_coll_mean=("mse_test_coll", "mean"),
            time_elapsed_mean=("time_elapsed", "mean"),
            n_runs=("file", "count"),
        )
        .reset_index()
    )
    return agg_df


def main():
    df = load_reg_search()
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


if __name__ == "__main__":
    main()
