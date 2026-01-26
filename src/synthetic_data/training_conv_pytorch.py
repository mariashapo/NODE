"""There no dedicated experiment runner for the synthetic Pytorch model, the same way there is one for Pyomo (PyomoExperimentRunner)."""
import argparse, os, time, pickle
from pathlib import Path
from typing import Any, Tuple
from utils.general import generate_seeds, str2bool
import argparse, json
from utils_training.run_train_toy import TrainerToy as Trainer


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


def _normalize_pretrain(pretrain: Any) -> Any:
    """Treat single full-length schedules as no-pretrain."""
    if _is_pyomo_pretrain(pretrain):
        return pretrain
    if pretrain in (None, False):
        return False
    # Collapse scalar >=1 or single-element list/tuple >=1 to False
    try:
        if isinstance(pretrain, (int, float)):
            return False if float(pretrain) >= 1 else [pretrain]
        if isinstance(pretrain, (list, tuple)) and len(pretrain) == 1:
            val = float(pretrain[0])
            return False if val >= 1 else pretrain
    except Exception:
        pass
    return pretrain


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


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/pytorch")
    p.add_argument("--data_type", default="ho")
    p.add_argument("--max_iter", type=json.loads, default=[400, 1000])
    p.add_argument("--pretrain", type=str, default="[0.2,1]")
    p.add_argument("--layer_width", type=json.loads, default=None)
    p.add_argument("--reg_norm", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--time_invariant", type=str2bool, default=True)
    p.add_argument("--penalty_lambda_reg", type=float, default=1e-3)
    p.add_argument("--noise_level", type=float, default=None)
    p.add_argument("--pyomo_bundle_dir", default="results/pyomo_pretrain", help="Where to look for pyomo pretrain bundles.")
    p.add_argument("--meta", action="store_true", default=True, help="Write run_meta.json with args/params.")
    return p


def parse_args(argv=None):
    return _build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pretrain_value = _normalize_pretrain(_parse_pretrain_arg(args.pretrain))
    
    params_model = {
        'layer_widths': args.layer_width if args.layer_width is not None else [2, 32, 2],
        'penalty_lambda_reg': args.penalty_lambda_reg,
        'time_invariant': args.time_invariant,
        'learning_rate': 1e-3,
        'max_iter': args.max_iter,
        'pretrain': pretrain_value,
        'split_time': True,
        'rtol': 1e-3,
        'atol': 1e-6,
        'reg_norm': args.reg_norm,
    }
    
    
    all_results = []
    run_date = time.strftime('%d%m%y')
    print("STARTING TRAINING")
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed}")
        params_for_seed = params_model.copy()
        custom_weights = None
        pyomo_bundle_time = None
        pyomo_bundle_name = None
        if _is_pyomo_pretrain(pretrain_value):
            bundle_path, bundle = _load_pyomo_bundle(pretrain_value, args.pyomo_bundle_dir, args.data_type, params_for_seed["layer_widths"])
            pyomo_bundle_time = _bundle_time(bundle)
            pyomo_bundle_name = str(bundle_path)
            custom_weights = bundle.get("weights_pytorch")
            if custom_weights is None:
                raise ValueError(f"No PyTorch weights found in bundle {bundle_path}")
            params_for_seed["pretrain"] = False  # skip fractional pretraining; use weights instead

        trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type = "pytorch", noise_level=args.noise_level)
        params_for_seed["log"] = True
        trainer.train(params_for_seed, custom_weights, seed = seed)
        results = trainer.extract_results_pytorch()
        results['train_loss'] = trainer.losses
        results['data_type'] = args.data_type
        results['pretrain'] = pretrain_value
        results['max_iter'] = args.max_iter
        # time should be measured off the model with no exta logging computations!!!
        trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type = "pytorch", noise_level=args.noise_level)
        params_for_seed["log"] = False
        trainer.train(params_for_seed, custom_weights, seed = seed)
        results_no_log = trainer.extract_results_pytorch()
        time_elapsed = results_no_log["time_elapsed"]
        if pyomo_bundle_time is not None:
            if isinstance(time_elapsed, list):
                time_elapsed = [pyomo_bundle_time] + time_elapsed
            elif time_elapsed is not None:
                time_elapsed = pyomo_bundle_time + time_elapsed
        results["time_elapsed"] = time_elapsed
        results["pyomo_pretraining"] = _is_pyomo_pretrain(pretrain_value)
        results["pyomo_pretraining_time"] = pyomo_bundle_time
        results["pyomo_bundle"] = pyomo_bundle_name
        all_results.append(results)

        ts = time.strftime('%Y-%m-%d_%H-%M')
        # create the top-level results directory if needed
        os.makedirs(args.outdir, exist_ok=True)

        max_iter = str(args.max_iter).strip('[]').replace(',','_').replace(' ','')
        # create a dated subfolder for this run
        subdir = os.path.join(args.outdir, f"pytorch_{args.data_type}_{max_iter}_{run_date}")
        os.makedirs(subdir, exist_ok=True)

        if args.meta:
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
                        },
                        f,
                        indent=2,
                    )

        # full filename for this seed
        filename = os.path.join(subdir, f"{seed}_{ts}.pkl")

        # write out the results for this seed
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
