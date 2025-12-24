"""Utility to train a Pyomo collocation model and persist weights for sequential pretraining."""
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from utils.general import str2bool
from utils_training.run_train_toy import TrainerToy
from utils_training.optimize_diffrax_rl import ExperimentRunner as DiffraxExperimentRunner
from utils_training.utils_pytorch import prepare_custom_weights


DEFAULT_CONFIG = Path("src/configs/config_pyomo_synth.json")


def _parse_json_list(val: str) -> Any:
    try:
        return json.loads(val)
    except Exception:
        return val


def _parse_skip_collocation(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("inf", "infinity"):
            return np.inf
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(config_path.read_text())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Pyomo and save weights for sequential pretraining.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Pyomo config to read defaults from.")
    p.add_argument("--outdir", default="results/pyomo_pretrain", help="Where to write the pretraining bundle.")
    p.add_argument("--data_type", choices=["vdp", "ho", "do"], default="vdp")
    p.add_argument("--layer_width", type=_parse_json_list, default=None, help="E.g. \"[2,32,2]\"; falls back to config.")
    p.add_argument("--penalty_lambda_reg", type=float, default=None, help="Override Pyomo weight decay.")
    p.add_argument("--tol", type=float, default=None, help="Override solver tolerance.")
    p.add_argument("--time_invariant", type=str2bool, default=None, help="Override time_invariant flag.")
    p.add_argument("--reg_norm", type=str2bool, default=None, help="Override reg_norm flag.")
    p.add_argument("--act_func", default=None, help="Activation function, e.g., tanh/softplus.")
    p.add_argument("--w_init_method", default=None, help="Weight init method (xavier/he/random).")
    p.add_argument("--skip_collocation", default=None, help="Skip collocation after N iterations (number or 'inf').")
    p.add_argument("--pre_initialize", type=str2bool, default=True, help="Whether to pre-initialize solution guess.")
    p.add_argument("--redirect_logs", type=str2bool, default=False, help="Redirect IPOPT logs to file.")
    p.add_argument("--spacing_type", default=None, help="Node spacing: chebyshev/gauss_legendre/gauss_radau/gauss_lobatto.")
    p.add_argument("--noise_level", type=float, default=None, help="Noise level for synthetic data.")
    p.add_argument("--seed", type=int, default=0)
    return p


def _build_model_params(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_cfg = cfg.get("model_params", {})
    solver_cfg = model_cfg.get("params", {})
    data_cfg = cfg.get("data", {})

    layer_widths = args.layer_width or model_cfg.get("layer_widths")
    penalty = args.penalty_lambda_reg if args.penalty_lambda_reg is not None else model_cfg.get("penalty_lambda_reg", 1e-2)
    tol = args.tol if args.tol is not None else solver_cfg.get("tol")
    spacing = args.spacing_type or data_cfg.get("spacing_type", "chebyshev")
    noise = args.noise_level if args.noise_level is not None else data_cfg.get("noise_level", TrainerToy._default_noise_level())

    solver_params = solver_cfg.copy()
    if tol is not None:
        solver_params["tol"] = tol

    model_params = {
        "layer_widths": layer_widths,
        "act_func": args.act_func or model_cfg.get("act_func", "tanh"),
        "penalty_lambda_reg": penalty,
        "time_invariant": model_cfg.get("time_invariant", True) if args.time_invariant is None else args.time_invariant,
        "w_init_method": args.w_init_method or model_cfg.get("w_init_method", "xavier"),
        "reg_norm": model_cfg.get("reg_norm", True) if args.reg_norm is None else args.reg_norm,
        "skip_collocation": _parse_skip_collocation(args.skip_collocation or model_cfg.get("skip_collocation", np.inf)),
        "redirect_logs": args.redirect_logs,
        "pre_initialize": args.pre_initialize,
        "params": solver_params,
    }

    return model_params, {"spacing_type": spacing, "noise_level": noise}


def _compute_mse(trainer: TrainerToy) -> Tuple[float, float]:
    pred_train = np.array(trainer.model.neural_ode(trainer.init_state, trainer.t))
    pred_test = np.array(trainer.model.neural_ode(trainer.init_state_test, trainer.t_test))
    mse_train = float(np.mean((np.array(trainer.y) - pred_train) ** 2))
    mse_test = float(np.mean((np.array(trainer.y_test) - pred_test) ** 2))
    return mse_train, mse_test


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    model_params, data_params = _build_model_params(args, cfg)

    spacing = data_params["spacing_type"]
    noise = data_params["noise_level"]
    trainer = TrainerToy.load_trainer(args.data_type, spacing_type=spacing, model_type="pyomo", noise_level=noise)

    start_wall = time.time()
    trainer.train_pyomo(model_params, seed=args.seed)
    wall_time = time.time() - start_wall

    weights_raw = trainer.extract_pyomo_weights()
    weights_jax = DiffraxExperimentRunner.format_weights_from_pyomo(weights_raw)
    weights_pt = prepare_custom_weights(weights_raw)
    mse_train, mse_test = _compute_mse(trainer)

    payload = {
        "weights_pyomo_raw": weights_raw,
        "weights_jax": weights_jax,
        "weights_pytorch": weights_pt,
        "metrics": {
            "mse_train": mse_train,
            "mse_test": mse_test,
        },
        "timing": {
            "solver_time": float(trainer.time_elapsed) if getattr(trainer, "time_elapsed", None) is not None else None,
            "wall_time": wall_time,
        },
        "termination": getattr(trainer, "termination", None),
        "args": vars(args),
        "data": {
            "spacing_type": spacing,
            "noise_level": noise,
        },
        "model_params": model_params,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    lw_tag = "-".join(str(w) for w in model_params["layer_widths"])
    fname = outdir / f"pyomo_pretrain_{args.data_type}_w{lw_tag}_seed{args.seed}_{stamp}.pkl"

    with open(fname, "wb") as f:
        pickle.dump(payload, f)

    meta_path = outdir / f"{fname.stem}.json"
    meta = {
        "bundle": fname.name,
        "timestamp": stamp,
        "data_type": args.data_type,
        "layer_widths": model_params["layer_widths"],
        "seed": args.seed,
        "timing": payload["timing"],
        "termination": payload["termination"],
        "metrics": payload["metrics"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[pyomo-pretrain] Saved bundle to {fname}")
    print(f"[pyomo-pretrain] Metrics: train={mse_train:.4e}, test={mse_test:.4e}")
    if payload["timing"]["solver_time"] is not None:
        print(f"[pyomo-pretrain] Solver time: {payload['timing']['solver_time']:.2f}s; wall time: {wall_time:.2f}s")


if __name__ == "__main__":
    main()
