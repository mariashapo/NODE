import argparse
import pickle
from pathlib import Path

import pandas as pd


def load_results(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def summarize_runs(results, source_name):
    rows = []
    for idx, run in enumerate(results):
        # Lists per iteration
        iters = run.get("iter", [])
        mse_train = run.get("mse_diffrax", [])
        mse_coll_train = run.get("mse_collocation_train", [])
        mse_test = run.get("mse_test_diffrax", [])
        mse_coll_test = run.get("mse_collocation_test", [])
        time_elapsed = run.get("time_elapsed", [])
        primal = run.get("primal_residual", [])

        seed_val = run.get("seed", None)

        max_len = max(
            len(iters),
            len(mse_train),
            len(mse_coll_train),
            len(mse_test),
            len(mse_coll_test),
            len(time_elapsed),
            len(primal),
        )

        for i in range(max_len):
            rows.append(
                {
                    "file": source_name,
                    "run": idx,
                    "iter": iters[i] if i < len(iters) else None,
                    "seed": seed_val,
                    "primal_residual": primal[i] if i < len(primal) else None,
                    "mse_train": mse_train[i] if i < len(mse_train) else None,
                    "mse_coll_train": mse_coll_train[i] if i < len(mse_coll_train) else None,
                    "mse_test": mse_test[i] if i < len(mse_test) else None,
                    "mse_coll_test": mse_coll_test[i] if i < len(mse_coll_test) else None,
                    "time_elapsed": time_elapsed[i] if i < len(time_elapsed) else None,
                }
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze ADMM result pickles.")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parents[0] / "results" / "admm_runs"),
        help="Directory containing admm_results_*.pkl files.",
    )
    parser.add_argument(
        "--file",
        default="admm_results_2026-01-11_17-05-05.pkl",
        help="Optional single filename (admm_results_*.pkl); when set, loaded from --dir.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="If set, also dump raw results after the summary dataframe.",
    )
    args = parser.parse_args(argv)

    rows_all = []
    raw_all = []

    outdir = Path(args.dir)
    if not outdir.exists():
        raise FileNotFoundError(f"Results directory not found: {outdir}")

    if args.file:
        pkl_path = outdir / args.file
        if not pkl_path.exists():
            raise FileNotFoundError(f"File not found: {pkl_path}")
        pkls = [pkl_path]
    else:
        pkls = sorted(outdir.glob("admm_results_*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No admm_results_*.pkl files found in {outdir}")

    for pkl in pkls:
        results = load_results(pkl)
        rows_all.extend(summarize_runs(results, pkl.name))
        if args.raw:
            raw_all.append({"file": pkl.name, "results": results})

    df = pd.DataFrame(rows_all)
    pd.set_option("display.max_columns", None)
    print(df)

    if args.raw:
        print("\n-- RAW RESULTS --")
        for entry in raw_all:
            print(f"\nFILE: {entry['file']}")
            print(entry["results"])

    return df


if __name__ == "__main__":
    main()
