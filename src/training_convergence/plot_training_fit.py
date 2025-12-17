"""Plot training data vs model predictions for a single synthetic run.

Runs one training job (Pyomo/JAX/PyTorch) using TrainerToy, extracts
predicted trajectories, and saves side-by-side train/test plots.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

# Headless-friendly backend for scripts
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils_training.run_train_toy import TrainerToy  # noqa: E402


def _to_np(x: Any) -> np.ndarray:
    """Convert torch/jax/np arrays to NumPy."""
    return np.asarray(x)


def _plot_combined(
    t_train,
    y_train,
    y_noisy,
    y_pred_train,
    t_test,
    y_test,
    y_pred_test,
    title,
    outpath: Path,
):
    """Plot all states with train+test on a single axes (common styling)."""
    t_train = _to_np(t_train)
    y_train = _to_np(y_train)
    y_noisy = _to_np(y_noisy)
    y_pred_train = _to_np(y_pred_train)

    t_test = _to_np(t_test)
    y_test = _to_np(y_test)
    y_pred_test = _to_np(y_pred_test)

    n_states = y_train.shape[1] if y_train.ndim > 1 else 1
    fig, ax = plt.subplots(figsize=(12, 6))

    # Shared styling per trace type to keep legend compact
    colors = {
        "train_clean": "C0",
        "train_noisy": "C1",
        "train_pred": "C3",
        "test_clean": "C0",
        "test_pred": "C3",
    }

    for i in range(n_states):
        yt = y_train[:, i] if n_states > 1 else y_train
        yn = y_noisy[:, i] if n_states > 1 else y_noisy
        yp_tr = y_pred_train[:, i] if n_states > 1 else y_pred_train
        ys = y_test[:, i] if n_states > 1 else y_test
        yp_te = y_pred_test[:, i] if n_states > 1 else y_pred_test

        lbl_train_clean = "train clean" if i == 0 else None
        lbl_train_noisy = "train noisy" if i == 0 else None
        lbl_train_pred = "train pred" if i == 0 else None
        lbl_test_clean = "test clean" if i == 0 else None
        lbl_test_pred = "test pred" if i == 0 else None

        ax.plot(t_train, yt, color=colors["train_clean"], linewidth=2.0, label=lbl_train_clean)
        ax.scatter(t_train, yn, color=colors["train_noisy"], s=10, alpha=0.45, label=lbl_train_noisy)
        ax.plot(t_train, yp_tr, color=colors["train_pred"], linewidth=2.0, linestyle="--", label=lbl_train_pred)

        ax.plot(t_test, ys, color=colors["test_clean"], linewidth=2.0, linestyle="-.", label=lbl_test_clean)
        ax.plot(t_test, yp_te, color=colors["test_pred"], linewidth=2.0, linestyle=":", label=lbl_test_pred)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    # drop None labels
    handles_labels = [(h, l) for h, l in zip(handles, labels) if l]
    uniq = dict(zip([l for _, l in handles_labels], [h for h, _ in handles_labels]))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, ncol=3)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


def _plot_split_compare(
    t_train,
    y_train,
    y_noisy,
    t_test,
    y_test,
    pred_map,
    title,
    outpath: Path,
    split_states: bool = False,
):
    """Plot train/test on separate axes with multiple model predictions overlaid."""
    t_train = _to_np(t_train)
    y_train = _to_np(y_train)
    y_noisy = _to_np(y_noisy)
    t_test = _to_np(t_test)
    if t_test.size > 0:
        t_test = t_test - t_test[0]
    y_test = _to_np(y_test)

    n_states = y_train.shape[1] if y_train.ndim > 1 else 1
    n_rows = n_states if split_states else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), sharey=False)
    axes = np.atleast_2d(axes)

    # base styling
    axes[0, 0].set_title("Train")
    axes[0, 1].set_title("Test")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    for r in range(n_rows):
        axes[r, 0].set_ylabel(f"state {r} value")
        axes[r, 1].set_ylabel(f"state {r} value")

    # Plot data once
    # Track global y-limits for optional syncing
    global_min, global_max = np.inf, -np.inf
    for i in range(n_states):
        row = i if split_states else 0
        yt = y_train[:, i] if n_states > 1 else y_train
        yn = y_noisy[:, i] if n_states > 1 else y_noisy
        ys = y_test[:, i] if n_states > 1 else y_test
        lbl_clean = "clean" if i == 0 else None
        lbl_noisy = "noisy" if i == 0 else None
        axes[row, 0].plot(t_train, yt, color="black", linewidth=2.0, label=lbl_clean)
        axes[row, 0].scatter(t_train, yn, color="gray", s=10, alpha=0.45, label=lbl_noisy)
        axes[row, 1].plot(t_test, ys, color="black", linewidth=2.0, label=lbl_clean)
        global_min = min(global_min, np.nanmin(yt), np.nanmin(yn), np.nanmin(ys))
        global_max = max(global_max, np.nanmax(yt), np.nanmax(yn), np.nanmax(ys))

    # Model colors
    model_colors = {
        "pyomo": "C0",
        "jax_diffrax": "C1",
        "pytorch": "C2",
    }

    for model, preds in pred_map.items():
        y_pred_train = _to_np(preds["y_pred_train"])
        y_pred_test = _to_np(preds["y_pred_test"])
        t_tr = _to_np(preds["t_train"])
        t_te = _to_np(preds["t_test"])
        # normalize test time to start at zero for visual alignment
        if t_te.size > 0:
            t_te = t_te - t_te[0]
        color = model_colors.get(model, None)
        label_tr = model.replace("jax_diffrax", "JAX").replace("pyomo", "Pyomo").replace("pytorch", "PyTorch")
        label_te = label_tr
        for i in range(n_states):
            row = i if split_states else 0
            yp_tr = y_pred_train[:, i] if n_states > 1 else y_pred_train
            yp_te = y_pred_test[:, i] if n_states > 1 else y_pred_test
            axes[row, 0].plot(t_tr, yp_tr, color=color, linestyle="--", linewidth=2.0, label=label_tr if i == 0 else None)
            axes[row, 1].plot(t_te, yp_te, color=color, linestyle="--", linewidth=2.0, label=label_te if i == 0 else None)
            global_min = min(global_min, np.nanmin(yp_tr), np.nanmin(yp_te))
            global_max = max(global_max, np.nanmax(yp_tr), np.nanmax(yp_te))

    legend_entries = {}
    for row in range(n_rows):
        for col in range(2):
            ax = axes[row, col]
            ax.grid(True, linestyle="--", alpha=0.4)
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                legend_entries[l] = h
            # sync y-limits across train/test
            if np.isfinite(global_min) and np.isfinite(global_max):
                pad = 0.05 * (global_max - global_min) if global_max > global_min else 0.1
                ax.set_ylim(global_min - pad, global_max + pad)

    # Single legend at bottom center
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


def parse_json(arg: str):
    try:
        return json.loads(arg)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Expected JSON, got {arg}") from exc


def build_parser():
    p = argparse.ArgumentParser(description="Plot synthetic training data vs predictions.")
    p.add_argument("--model_type", choices=["pyomo", "jax_diffrax", "pytorch"], default="pyomo", help="Single model type (ignored if --model_types is provided).")
    p.add_argument("--model_types", nargs="+", choices=["pyomo", "jax_diffrax", "pytorch"], help="Optional list of models to compare/overlay.")
    p.add_argument("--data_type", choices=["ho", "vdp", "do"], default="ho")
    p.add_argument("--layer_width", type=parse_json, default=None, help="JSON list, e.g. '[2,32,2]'")
    p.add_argument("--max_iter", type=parse_json, default=None, help="Optional JSON list; per-model defaults are used when omitted")
    p.add_argument("--pretrain", type=parse_json, default="[0.2,1]", help="JSON list of fractions or []/false for none")
    p.add_argument("--penalty_lambda_reg", type=float, default=0.01, help="Regularization for Pyomo/JAX/PT")
    p.add_argument("--reg_norm", action=argparse.BooleanOptionalAction, default=True, help="Normalize L2 regularization by parameter count (default True to match Pyomo).")
    p.add_argument("--tol", type=float, default=1e-12, help="IPOPT tol (Pyomo only)")
    p.add_argument("--noise_level", type=float, default=None, help="Override noise level; if omitted, uses config value.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("results/plots"))
    p.add_argument("--plot_mode", choices=["combined", "split_compare"], default="combined", help="combined: single model with train+test on one axes; split_compare: train/test split with multiple models overlaid.")
    return p


def make_pyomo_params(args) -> Dict[str, Any]:
    return {
        "layer_widths": args.layer_width if args.layer_width is not None else [2, 32, 2],
        "act_func": "tanh",
        "penalty_lambda_reg": args.penalty_lambda_reg,
        "time_invariant": True,
        "w_init_method": "xavier",
        "reg_norm": args.reg_norm,
        "skip_collocation": np.inf,
        "redirect_logs": False,
        "params": {
            "tol": args.tol,
            "print_level": 3,
            "max_iter": 6000,
            "constr_viol_tol": args.tol,
            "compl_inf_tol": args.tol,
            "dual_inf_tol": args.tol,
            "acceptable_tol": 1e-9,
            "acceptable_constr_viol_tol": 1e-9,
        },
    }


def make_jax_pt_params(args, model_type: str) -> Dict[str, Any]:
    default_max_iter = {
        "jax_diffrax": [500, 10000],
        "pytorch": [400, 1000],
    }
    max_iter = args.max_iter
    if max_iter is None:
        max_iter = default_max_iter.get(model_type, [400, 1000])

    return {
        "layer_widths": args.layer_width if args.layer_width is not None else [2, 32, 2],
        "penalty_lambda_reg": args.penalty_lambda_reg,
        "reg_norm": args.reg_norm,
        "time_invariant": True,
        "learning_rate": 1e-3,
        "max_iter": max_iter if isinstance(max_iter, Sequence) else [max_iter],
        "pretrain": args.pretrain if args.pretrain not in (False, None, []) else False,
        "split_time": True,
        "rtol": 1e-3,
        "atol": 1e-6,
        "act_func": "tanh",
    }


def main():
    args = build_parser().parse_args()

    model_list = args.model_types if getattr(args, "model_types", None) else [args.model_type]
    base_trainer = None
    preds = {}

    for m in model_list:
        spacing = "chebyshev" if m == "pyomo" else "uniform"
        trainer = TrainerToy.load_trainer(
            args.data_type,
            spacing_type=spacing,
            model_type=m,
            noise_level=args.noise_level,
        )
        if base_trainer is None:
            base_trainer = trainer  # capture data/ts from the first model

        if m == "pyomo":
            params_model = make_pyomo_params(args)
            trainer.train_pyomo(params_model, seed=args.seed)
            results = trainer.extract_results_pyomo(detailed=True)
            print("Pyomo: ", results["mse_train"], results["mse_test"])
            y_pred_train = results["odeint_pred"]
            y_pred_test = results["odeint_pred_test"]
        elif m == "jax_diffrax":
            params_model = make_jax_pt_params(args, m)
            trainer.train_diffrax(params_model, custom_params=None, seed=args.seed)
            results = trainer.extract_results_diffrax(detailed=True)
            print("Jax Diffrax: ", results["mse_train"], results["mse_test"])
            y_pred_train = results["odeint_pred"]
            y_pred_test = results["odeint_pred_test"]
        else:
            params_model = make_jax_pt_params(args, m)
            trainer.train_pytorch(params_model, custom_params=None, seed=args.seed)
            results = trainer.extract_results_pytorch(detailed=True)
            print("PyTorch: ", results["mse_train"], results["mse_test"])
            y_pred_train = results["odeint_pred"]
            y_pred_test = results["odeint_pred_test"]

        preds[m] = {
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "t_train": trainer.t,
            "t_test": trainer.t_test,
        }

    title = f"{'/'.join(model_list)} — {args.data_type} — seed {args.seed}"
    outfile = args.outdir / f"{'-'.join(model_list)}_{args.data_type}_seed{args.seed}.png"

    if args.plot_mode == "split_compare" or len(model_list) > 1:
        _plot_split_compare(
            base_trainer.t,
            base_trainer.y,
            base_trainer.y_noisy,
            base_trainer.t_test,
            base_trainer.y_test,
            preds,
            title,
            outfile,
            split_states=(args.plot_mode == "split_compare"),
        )
    else:
        y_pred_train, y_pred_test = preds[model_list[0]]
        _plot_combined(
            base_trainer.t,
            base_trainer.y,
            base_trainer.y_noisy,
            y_pred_train,
            base_trainer.t_test,
            base_trainer.y_test,
            y_pred_test,
            title,
            outfile,
        )


if __name__ == "__main__":
    main()
