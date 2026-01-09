import argparse
import logging
import pickle
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR, filename='error_log.txt')

from utils_training.optimize_diffrax_rl import ExperimentRunner


def parse_list(value, cast):
    """Parse comma-separated values into a list with the desired type."""
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    parts = [p.strip() for p in str(value).split(',') if p.strip()]
    return [cast(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Run JAX/diffrax experiments.")
    parser.add_argument(
        "--num-epochs",
        default="1000,10000",
        help="Comma-separated epochs per stage (e.g. 100,10).",
    )
    parser.add_argument(
        "--pretrain",
        default="0.2,1",
        help="Comma-separated fractions of data per stage (e.g. 0.2,1).",
    )
    args = parser.parse_args()

    # repo root is two levels up from src/real_life_data/
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = REPO_ROOT / "data" / "df_train.csv"
    OUTDIR = REPO_ROOT / "results" / "jax_rl_runs"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    start_date = "2015-01-15"

    num_epochs = parse_list(args.num_epochs, int)
    pretrain = parse_list(args.pretrain, float)

    extra_inputs = {}
    extra_inputs["params_model"] = {
        "layer_sizes": [7, 32, 1],
        "penalty": 1e-5,
        "learning_rate": 1e-2,
        "num_epochs": num_epochs,
        "pretrain": pretrain,
    }

    extra_inputs["params_data"] = {
        "file_path": str(DATA_PATH),
        "start_date": start_date,
        "n_points": 300,
        "split": 200,
        "n_days": 1,
        "m": 1,
        "prev_hour": False,
        "prev_week": True,
        "prev_year": True,
        "spacing": "uniform",
        "encoding": {
            "settlement_date": "t",
            "temperature": "var1",
            "hour": "var2",
            "nd": "y",
        },
    }

    extra_inputs["params_sequence"] = {"sequence_len": 15, "frequency": 3}
    extra_inputs["params_results"] = {"plot": True, "log": 50, "split_time": True}
    # extra_inputs['trained_wb'] = trained_wb

    runner = ExperimentRunner(start_date, "default", extra_inputs)
    runner.run()

    # Persist results similarly to training_convergence style
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    subdir = OUTDIR / f"jax_rl_{ts}"
    subdir.mkdir(parents=True, exist_ok=True)

    with (subdir / "results_WITH_logging.pkl").open("wb") as f:
        pickle.dump(runner.results_full, f)

    print(f"Saved results to {subdir}")

    # rerun the same but no logging
    extra_inputs["params_results"] = {"plot": True, "log": False, "split_time": True}

    runner = ExperimentRunner(start_date, "default", extra_inputs)
    runner.run()

    with (subdir / "results_NO_logging.pkl").open("wb") as f:
        pickle.dump(runner.results_full, f)

    print(f"Saved results to {subdir}")


if __name__ == "__main__":
    main()
