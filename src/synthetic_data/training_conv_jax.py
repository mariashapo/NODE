"""There no dedicated experiment runner for the synthetic Jax-diffrax model, the same way there is one for Pyomo (PyomoExperimentRunner)."""
import argparse, os, time, pickle
from pathlib import Path
from typing import Any, Tuple
from utils.general import generate_seeds, print_memory, str2bool
import argparse, json
from utils_training.run_train_toy import TrainerToy as Trainer
import gc, ctypes
import jax, gc, ctypes


def _parse_pretrain_arg(val: Any) -> Any:
    """Accept JSON lists or literal strings like 'pyomo'/'pyomo:/path'."""
    if val is None:
        return [0.2, 1]
    if isinstance(val, (list, tuple)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val


def _is_pyomo_pretrain(pretrain: Any) -> bool:
    return isinstance(pretrain, str) and pretrain.startswith("pyomo")


def _format_width_tag(layer_widths) -> str:
    return "-".join(str(w) for w in layer_widths)


def _find_latest_bundle(bundle_dir: Path, data_type: str, layer_widths) -> Path:
    pattern = f"pyomo_pretrain_{data_type}_w{_format_width_tag(layer_widths)}_*.pkl"
    candidates = sorted(bundle_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_pyomo_bundle(pretrain_spec: str, bundle_dir: str, data_type: str, layer_widths) -> Tuple[Path, dict]:
    bundle_path = None
    if ":" in pretrain_spec:
        _, path_str = pretrain_spec.split(":", 1)
        path_candidate = Path(path_str).expanduser()
        if path_candidate.is_file():
            bundle_path = path_candidate
        else:
            # treat as a tag/prefix inside bundle_dir, pick latest match
            tag = path_candidate.name
            pattern = f"{tag}*.pkl"
            candidates = sorted(Path(bundle_dir).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            bundle_path = candidates[0] if candidates else None
    else:
        bundle_path = _find_latest_bundle(Path(bundle_dir), data_type, layer_widths)
    if not bundle_path or not bundle_path.exists():
        raise FileNotFoundError(f"Pyomo bundle not found for spec '{pretrain_spec}'. Looked in {bundle_dir}.")
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle_path, bundle


def _bundle_time(bundle: dict) -> float:
    timing = bundle.get("timing", {}) if bundle else {}
    return timing.get("wall_time") or timing.get("solver_time")


def _cleanup_trainer(tr):
    if tr is None:
        return
    # If your Trainer exposes a closer, call it
    close = getattr(tr, "close", None)
    if callable(close):
        try: close()
        except Exception: pass
    # Null out common big fields (adjust to your Trainer)
    for attr in (
        "state","model","params","apply_fn","history","losses",
        "train_data","test_data","X","Y","ts","ys","solution"
    ):
        if hasattr(tr, attr):
            try: setattr(tr, attr, None)
            except Exception: pass

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/jax")
    p.add_argument("--data_type", default = "ho")
    p.add_argument("--max_iter", type=json.loads, default=[200, 200])
    p.add_argument("--pretrain", type=str, default="[0.2,1]")
    p.add_argument("--log", type=json.loads, default=100)
    p.add_argument("--timing_only", type=str2bool, default=False, help="Skip the logging run; run only the timing pass.")
    p.add_argument("--layer_width", type=json.loads, default=None)
    p.add_argument("--reg_norm", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--time_invariant", type=str2bool, default=True)
    p.add_argument("--penalty_lambda_reg", type=float, default=1e-3)
    p.add_argument("--noise_level", type=float, default=None)
    p.add_argument("--pyomo_bundle_dir", default="results/pyomo_pretrain", help="Where to look for pyomo pretrain bundles.")
    return p

def parse_args(argv=None):
    return _build_parser().parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    pretrain_value = _parse_pretrain_arg(args.pretrain)
    
    params_model = {
        'layer_widths': args.layer_width if args.layer_width is not None else [2, 32, 2],
        'penalty_lambda_reg': args.penalty_lambda_reg,
        'time_invariant': args.time_invariant,
        'learning_rate': 1e-3,
        'max_iter': args.max_iter,
        'pretrain': pretrain_value,
        'log': args.log,
        'reg_norm': args.reg_norm,
        'split_time': True,
        'rtol': 1e-3,
        'atol': 1e-6,
        'act_func': 'tanh',
    }
    
    run_time = time.strftime("%d%m%y_%H%M")
    layer_widths = args.layer_width if args.layer_width is not None else params_model["layer_widths"]
    width_tag = _format_width_tag(layer_widths) if layer_widths is not None else "unknown"
    # all_results = []
    print("STARTING TRAINING")
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed}")
        results = {}
        trainer = None
        custom_params = None
        pyomo_bundle_time = None
        pyomo_bundle_name = None

        params_for_seed = params_model.copy()
        # Handle Pyomo pretraining bundle injection
        if _is_pyomo_pretrain(pretrain_value):
            bundle_path, bundle = _load_pyomo_bundle(pretrain_value, args.pyomo_bundle_dir, args.data_type, params_for_seed["layer_widths"])
            pyomo_bundle_time = _bundle_time(bundle)
            pyomo_bundle_name = str(bundle_path)
            custom_params = bundle.get("weights_jax")
            if custom_params is None:
                raise ValueError(f"No JAX weights found in bundle {bundle_path}")
            params_for_seed["pretrain"] = False  # skip fractional pretraining; use weights instead
        print_memory("Memory use loop start: ")
        try:
            if (not args.timing_only) and args.log > 0:
                trainer = Trainer.load_trainer(
                    args.data_type,
                    spacing_type="uniform",
                    model_type="jax_diffrax",
                    noise_level=args.noise_level,
                )
                params_for_seed["log"] = args.log
                trainer.train(params_for_seed, custom_params, seed=seed)
                results = trainer.extract_results() or {}
                results["train_loss"] = getattr(trainer, "losses", None)
                results["data_type"] = args.data_type
                results["pretrain"] = pretrain_value
                results["max_iter"] = args.max_iter
        finally:
            print_memory("Memory before trainer cleanup: ")
            _cleanup_trainer(trainer)
            del trainer
            jax.clear_caches()
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass
        print_memory("Memory after trainer use: ")
        # ---- timing run (no per-epoch logging logging) ----
        try:
            trainer = Trainer.load_trainer(
                args.data_type,
                spacing_type="uniform",
                model_type="jax_diffrax",
                noise_level=args.noise_level,
            )
            params_for_seed["log"] = False
            trainer.train(params_for_seed, custom_params, seed=seed)
            results_no_log = trainer.extract_results() or {}
        finally:
            _cleanup_trainer(trainer)
            del trainer
            jax.clear_caches()
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass

        if args.timing_only or not args.log:
            results = results_no_log
            results["train_loss"] = results.get("train_loss")
            results["data_type"] = args.data_type
            results["pretrain"] = pretrain_value
            results["max_iter"] = args.max_iter
            
        # merge timing into results
        time_elapsed = results_no_log.get("time_elapsed")
        if pyomo_bundle_time is not None:
            if isinstance(time_elapsed, list):
                time_elapsed = [pyomo_bundle_time] + time_elapsed
            elif time_elapsed is not None:
                time_elapsed = pyomo_bundle_time + time_elapsed
        results["time_elapsed"] = time_elapsed
        results["pyomo_pretraining"] = _is_pyomo_pretrain(pretrain_value)
        results["pyomo_pretraining_time"] = pyomo_bundle_time
        results["pyomo_bundle"] = pyomo_bundle_name
        results["layer_widths"] = layer_widths
        results["penalty_lambda_reg"] = params_model.get("penalty_lambda_reg")
        results["reg_norm"] = params_model.get("reg_norm")
        print_memory("Current memory use: ")
        
        ts = time.strftime('%Y-%m-%d_%H-%M')
        # Create the top-level results directory if needed
        os.makedirs(args.outdir, exist_ok=True)

        max_iter = str(args.max_iter).strip('[]').replace(',','_').replace(' ','')
        # Create a dated subfolder for this run
        subdir = os.path.join(args.outdir, f"jax_{args.data_type}_w{width_tag}_{max_iter}_{run_time}")
        os.makedirs(subdir, exist_ok=True)

        # Persist run metadata once per subdir for traceability
        meta_path = os.path.join(subdir, "run_meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "args": vars(args),
                        "params_model": params_model,
                        "timestamp": ts,
                        "pyomo_pretraining": _is_pyomo_pretrain(pretrain_value),
                        "pyomo_bundle": pyomo_bundle_name,
                        "pyomo_pretraining_time": pyomo_bundle_time,
                        "layer_width": layer_widths,
                        "penalty_lambda_reg": params_model.get("penalty_lambda_reg"),
                        "reg_norm": params_model.get("reg_norm"),
                    },
                    f,
                    indent=2,
                )

        # Full filename for this seed
        filename = os.path.join(subdir, f"{seed}_{ts}.pkl")

        # Write out the results for this seed
        with open(filename, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {filename}")


if __name__ == "__main__":
    action = "prod"
    if action == "dev":
        main([
            "--data_type", "do",
            "--layer_width", "[3,8,2]",
            "--penalty_lambda_reg", "0.1",
            "--n_seeds", "1",
            "--time_invariant", "False",
            "--outdir", "results/study_do",
            "--max_iter", "[200,1000]",
            "--pretrain", "[0.2,1]",
        ])
    else:
        main()
