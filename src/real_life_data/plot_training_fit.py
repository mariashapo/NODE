"""Plot train/test trajectories for real-life data using the PyTorch Neural ODE."""

import argparse
from pathlib import Path

import matplotlib

# Headless-friendly backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from models.nn_pytorch import NeuralODE
from utils.preprocess import DataPreprocessor


def parse_list(value, cast):
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [cast(p) for p in parts]


def run_training(
    start_date: str,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    file_path: Path,
    encoding: dict,
    n_points: int,
    split: int,
    m: int,
    prev_hour: bool,
    prev_week: bool,
    prev_year: bool,
):
    """Load data, train PyTorch NeuralODE, return predictions and data."""
    data_loader = DataPreprocessor(
        str(file_path),
        start_date=start_date,
        number_of_points=n_points,
        n_days=1,
        m=m,
        feature_encoding=encoding,
        split=split,
        smooth=False,
        num_nodes_mult=1,
        prev_hour=prev_hour,
        prev_week=prev_week,
        prev_year=prev_year,
    )

    data_subsample = data_loader.load_data()
    df_train, df_test = data_loader.preprocess_data(data_subsample)

    ys_np = df_train["y"].to_numpy()
    ts_np = df_train["t"].to_numpy().reshape(-1)
    Xs_np = df_train.drop(columns=["y", "t"]).to_numpy()

    ys_test_np = df_test["y"].to_numpy()
    ts_test_np = df_test["t"].to_numpy().reshape(-1)
    Xs_test_np = df_test.drop(columns=["y", "t"]).to_numpy()

    # tensors
    y0 = np.array([ys_np[0]], dtype=np.float32)
    ys = np.atleast_2d(ys_np).T.astype(np.float32)
    ts = np.array(ts_np, dtype=np.float32).reshape(-1)

    y0_test = np.array([ys_test_np[0]], dtype=np.float32)
    ys_test = np.atleast_2d(ys_test_np).T.astype(np.float32)
    ts_test = np.array(ts_test_np, dtype=np.float32).reshape(-1)

    # Train single-stage
    layer_widths = [Xs_np.shape[1] + 1, 32, 1]  # +1 for state
    ode_model = NeuralODE(layer_widths, learning_rate, weight_decay=weight_decay, time_invariant=True)
    ode_model.train_model(
        ts,
        ys,
        y0,
        num_epochs=num_epochs,
        rtol=1e-3,
        atol=1e-4,
        extra_inputs=(
            np.concatenate([ts_np, ts_test_np]).reshape(-1),  # t_all
            np.concatenate([Xs_np, Xs_test_np]),              # extra inputs aligned with t_all
        ),
        verbose=False,
        log=None,
    )

    y_pred_train = ode_model.predict(ts, y0, extra_inputs=Xs_np)
    y_pred_test = ode_model.predict(ts_test, y0_test, extra_inputs=Xs_test_np)

    return dict(
        t_train=ts_np,
        y_train=ys_np,
        t_test=ts_test_np,
        y_test=ys_test_np,
        y_pred_train=np.squeeze(y_pred_train),
        y_pred_test=np.squeeze(y_pred_test),
    )


def plot_combined(data: dict, outpath: Path):
    """Plot train+test on one axes with unified legend."""
    t_train = np.asarray(data["t_train"])
    y_train = np.asarray(data["y_train"])
    t_test = np.asarray(data["t_test"])
    y_test = np.asarray(data["y_test"])
    y_pred_train = np.asarray(data["y_pred_train"])
    y_pred_test = np.asarray(data["y_pred_test"])

    fig, ax = plt.subplots(figsize=(10, 6))

    # True data (dotted), predictions (solid), observations as points
    ax.plot(t_train, y_train, color="C0", linestyle=":", linewidth=2.0, label="True Data")
    ax.plot(t_test, y_test, color="C0", linestyle=":", linewidth=2.0)
    ax.scatter(t_train, y_train, color="C1", s=40, alpha=0.6, label="Noisy Data")
    ax.plot(t_train, y_pred_train, color="C3", linewidth=2.0, label="Predicted Trajectory")
    ax.plot(t_test, y_pred_test, color="C3", linewidth=2.0)

    ax.set_xlabel("Time (t)", fontsize=16)
    ax.set_ylabel("State Values - u(t), v(t)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=13)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


def build_parser():
    p = argparse.ArgumentParser(description="Plot train/test trajectories for real-life data (PyTorch).")
    p.add_argument("--start-date", default="2015-01-15")
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=1e-1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--n-points", type=int, default=300)
    p.add_argument("--split", type=int, default=200)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--prev-hour", action="store_true", default=False)
    p.add_argument("--prev-week", action="store_true", default=True)
    p.add_argument("--prev-year", action="store_true", default=True)
    p.add_argument("--out", default="results/plots/rl_traj.png")
    return p


def main():
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / "data" / "df_train.csv"
    encoding = {"settlement_date": "t", "temperature": "var1", "hour": "var2", "nd": "y"}

    data = run_training(
        start_date=args.start_date,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        file_path=file_path,
        encoding=encoding,
        n_points=args.n_points,
        split=args.split,
        m=args.m,
        prev_hour=args.prev_hour,
        prev_week=args.prev_week,
        prev_year=args.prev_year,
    )

    plot_combined(data, Path(args.out))


if __name__ == "__main__":
    main()
