import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from models.nn_pytorch import NeuralODE
from utils.preprocess import DataPreprocessor
from utils_training.optimize_diffrax_rl import ExperimentRunner
from utils_training.utils_pytorch import prepare_custom_weights


def parse_list(value, cast):
    """Parse comma-separated values into a list with the desired type."""
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [cast(p) for p in parts]


def mse_numpy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute MSE, returning a python float."""
    p = pred.detach().cpu().numpy().flatten()
    t = target.detach().cpu().numpy().flatten()
    return float(np.mean((p - t) ** 2))


def to_torch_1d(x) -> torch.Tensor:
    return torch.tensor(np.array(x), dtype=torch.float32)


def to_torch_col(x) -> torch.Tensor:
    return torch.tensor(np.atleast_2d(x).T, dtype=torch.float32)


def load_weights_for_date(weights_dir: Path, date_str: str) -> Tuple[Optional[list], Optional[list]]:
    """Return (layer_sizes, weights) if a weights file for the date exists; otherwise (None, None)."""
    candidates = sorted(weights_dir.glob(f"*{date_str}*.pkl"))
    if not candidates:
        return None, None
    with candidates[-1].open("rb") as f:
        payload = pickle.load(f)
    return payload.get("layer_sizes"), payload.get("weights")


def run_staged_training(
    ode_model: NeuralODE,
    ts: torch.Tensor,
    ys: torch.Tensor,
    y0: torch.Tensor,
    pretrain_fracs: list[float],
    num_epochs: list[int],
    t_all: np.ndarray,
    Xs_full: np.ndarray,
    log,
):
    """Run staged training and return per-stage wall times."""
    times_elapsed = []
    n = len(ts)

    for frac, epochs in zip(pretrain_fracs, num_epochs):
        k = int(n * frac)
        k = max(1, min(k, n))  # guard against empty/out-of-range slices

        start_time = time.time()
        ode_model.train_model(
            ts[:k],
            ys[:k],
            y0,
            num_epochs=epochs,
            rtol=1e-3,
            atol=1e-6,
            extra_inputs=(t_all, Xs_full),
            log=log if (log and frac == pretrain_fracs[-1]) else False,
        )
        times_elapsed.append(time.time() - start_time)

    return times_elapsed


def main():
    repo_root = Path(__file__).resolve().parents[2]
    outdir = repo_root / "results" / "pytorch_rl_pyomo_weights"
    outdir.mkdir(parents=True, exist_ok=True)
    weights_dir = repo_root / "results" / "trained_wb"

    parser = argparse.ArgumentParser(description="Train PyTorch Neural ODE on real-life data.")
    parser.add_argument(
        "--pretrain",
        default="1",
        help="Comma-separated fractions of the data to use per training stage (e.g. 0.2,1).",
    )
    parser.add_argument(
        "--num-epochs",
        default="500",
        help="Comma-separated epochs per stage (e.g. 400,1000).",
    )
    parser.add_argument(
        "--skip-logging-run",
        action="store_true",
        help="Skip the first run with logging enabled (default: run both logged and no-log passes).",
    )
    parser.add_argument(
        "--use-pyomo-weights",
        action="store_true",
        default=False,
        help="If set, try to load Pyomo-trained weights matching each date. Default: disabled.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated list of weight files to load (one per date, overrides --use-pyomo-weights for those dates).",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="Comma-separated list of dates (YYYY-MM-DD). If provided, overrides automatic date generation.",
    )
    
    parser.add_argument(
        "--sequence_len",
        default=30,
    )
    args = parser.parse_args()

    pretrain_fracs = parse_list(args.pretrain, float)
    num_epochs = parse_list(args.num_epochs, int)
    if len(pretrain_fracs) != len(num_epochs):
        raise ValueError("Number of pretrain fractions must match number of epoch values.")

    file_path = repo_root / "data" / "df_train.csv"
    encoding = {"settlement_date": "t", "temperature": "var1", "hour": "var2", "nd": "y"}

    freq = 3
    sequence_len = int(args.sequence_len)
    start_date_str = "2015-01-15"
    _ = datetime.strptime(start_date_str, "%Y-%m-%d")  # keep for sanity check / future use

    layer_widths = [7, 32, 1]
    learning_rate = 1e-1
    weight_decay = 1e-5

    experiment_results_with_logs: dict = {}
    experiment_results_no_logs: dict = {}

    if args.dates:
        date_sequences = parse_list(args.dates, str)
    else:
        date_sequences = ExperimentRunner.generate_dates(start_date_str, sequence_len, freq)

    weights_paths = None
    if args.weights:
        weights_paths = [Path(p).expanduser() for p in parse_list(args.weights, str)]
        if len(weights_paths) != len(date_sequences):
            raise ValueError("Number of weight files must match number of dates.")

    # Create one output folder for the whole run (instead of one per date)
    run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    subdir = outdir / f"pytorch_rl_{run_stamp}"
    subdir.mkdir(parents=True, exist_ok=True)

    for i, date in enumerate(date_sequences):
        data_loader = DataPreprocessor(
            str(file_path),
            start_date=date,
            number_of_points=300,
            n_days=1,
            m=1,
            feature_encoding=encoding,
            split=200,
            smooth=False,
            num_nodes_mult=1,
            prev_hour=False,
            prev_week=True,
            prev_year=True,
        )

        data_subsample = data_loader.load_data()
        df_train, df_test = data_loader.preprocess_data(data_subsample)

        ys_np = df_train["y"].to_numpy()
        ts_np = df_train["t"].to_numpy()
        Xs_np = df_train.drop(columns=["y", "t"]).to_numpy()

        ys_test_np = df_test["y"].to_numpy()
        ts_test_np = df_test["t"].to_numpy()
        Xs_test_np = df_test.drop(columns=["y", "t"]).to_numpy()

        # extra inputs
        Xs_full = np.atleast_2d(np.concatenate([Xs_np, Xs_test_np]))
        t_all = np.concatenate([ts_np, ts_test_np])

        # torch tensors
        y0 = to_torch_1d([ys_np[0]])
        ys = to_torch_col(ys_np)
        ts = to_torch_1d(ts_np)

        y0_test = to_torch_1d([ys_test_np[0]])
        ys_test = to_torch_col(ys_test_np)
        ts_test = to_torch_1d(ts_test_np)

        # Load Pyomo-trained weights for this date if available
        lw_loaded, weights_loaded = (None, None)
        # Priority: explicit weight files > auto-discovery
        if weights_paths is not None:
            with weights_paths[i].open("rb") as f:
                payload = pickle.load(f)
            lw_loaded = payload.get("layer_sizes")
            weights_loaded = payload.get("weights")
        elif args.use_pyomo_weights:
            lw_loaded, weights_loaded = load_weights_for_date(weights_dir, date)
        layer_widths_use = lw_loaded if lw_loaded else layer_widths
        prepared_weights = prepare_custom_weights(weights_loaded) if weights_loaded else None

        if not args.skip_logging_run:
            # -------------------------
            # RUN 1: WITH LOGGING
            # -------------------------
            ode_model = NeuralODE(
                layer_widths_use,
                learning_rate,
                weight_decay=weight_decay,
                time_invariant=True,
                custom_weights=prepared_weights,
            )

            log = {
                "t": ts,
                "y": ys,
                "y_init": y0,
                "extra_args": None,  # extra inputs are initialized in the training
                "epoch_recording_step": 15,
                "t_test": ts_test,
                "y_test": ys_test,
                "y_init_test": y0_test,
                "extra_args_test": None,
            }

            times_elapsed = run_staged_training(
                ode_model=ode_model,
                ts=ts,
                ys=ys,
                y0=y0,
                pretrain_fracs=pretrain_fracs,
                num_epochs=num_epochs,
                t_all=t_all,
                Xs_full=Xs_full,
                log=log,
            )

            y_pred = ode_model.predict(ts, y0, extra_inputs=Xs_np)
            y_pred_test = ode_model.predict(ts_test, y0_test, extra_inputs=Xs_test_np)

            experiment_results_with_logs[date] = {
                "mse_train": mse_numpy(y_pred, ys),
                "mse_test": mse_numpy(y_pred_test, ys_test),
                "time_elapsed": float(np.sum(times_elapsed)),
                "time_elapsed_split": times_elapsed,
                "training_losses": ode_model.losses,
            }

        # -------------------------
        # RUN 2: NO LOGGING
        # -------------------------
        ode_model = NeuralODE(
            layer_widths_use,
            learning_rate,
            weight_decay=weight_decay,
            time_invariant=True,
            custom_weights=prepared_weights,
        )

        times_elapsed = run_staged_training(
            ode_model=ode_model,
            ts=ts,
            ys=ys,
            y0=y0,
            pretrain_fracs=pretrain_fracs,
            num_epochs=num_epochs,
            t_all=t_all,
            Xs_full=Xs_full,
            log=False,
        )

        y_pred = ode_model.predict(ts, y0, extra_inputs=Xs_np)
        y_pred_test = ode_model.predict(ts_test, y0_test, extra_inputs=Xs_test_np)

        experiment_results_no_logs[date] = {
            "mse_train": mse_numpy(y_pred, ys),
            "mse_test": mse_numpy(y_pred_test, ys_test),
            "time_elapsed": float(np.sum(times_elapsed)),
            "time_elapsed_split": times_elapsed,
            "training_losses": ode_model.losses,
        }
        
        print(f"Times elapsed for date {date} (no logging): {times_elapsed}")

    if not args.skip_logging_run:
        with (subdir / "results_WITH_logging.pkl").open("wb") as f:
            pickle.dump(experiment_results_with_logs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved WITH logging results to {subdir}")

    with (subdir / "results_NO_logging.pkl").open("wb") as f:
        pickle.dump(experiment_results_no_logs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved NO logging results to {subdir}")


if __name__ == "__main__":
    main()
