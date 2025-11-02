"""There no dedicated experiment runner for the synthetic Jax-diffrax model, the same way there is one for Pyomo (PyomoExperimentRunner)."""
import argparse, os, time, pickle
from utils.general import generate_seeds
import argparse, json
from utils_training.run_train_toy import TrainerToy as Trainer
import gc, ctypes


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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/jax")
    p.add_argument("--data_type", default = "ho")
    p.add_argument("--max_iter", type=json.loads, default=[200, 200])
    p.add_argument("--pretrain", type=json.loads, default=[0.2, 1])
    p.add_argument("--log", type=json.loads, default = 25)
    args = p.parse_args()
    
    params_model = {
        'layer_widths': [2, 32, 2],
        'penalty_lambda_reg': 1e-3,
        'time_invariant': True,
        'learning_rate': 1e-3,
        'max_iter': args.max_iter,
        'pretrain': args.pretrain,
        'log': args.log,
        'split_time': True,
        'rtol': 1e-3,
        'atol': 1e-6,
        'act_func': 'tanh',
    }
    
    all_results = []
    print("STARTING TRAINING")
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed}")

        results = {}
        trainer = None

        try:
            if args.log > 0:
                trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type="jax_diffrax")
                params_model["log"] = args.log
                trainer.train(params_model, seed=seed)
                results = trainer.extract_results() or {}
                results["train_loss"] = getattr(trainer, "losses", None)
                results["data_type"] = args.data_type
                results["pretrain"] = args.pretrain
                results["max_iter"] = args.max_iter
        finally:
            _cleanup_trainer(trainer)
            trainer = None
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass

        # ---- timing run (no per-epoch logging logging) ----
        try:
            trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type="jax_diffrax")
            params_model["log"] = False
            trainer.train(params_model, seed=seed)
            results_no_log = trainer.extract_results() or {}
        finally:
            _cleanup_trainer(trainer)
            trainer = None
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass

        # merge timing into results
        if not args.log:
            results["train_loss"] = results_no_log.get("train_loss", getattr(trainer, "losses", None))
            results["data_type"] = args.data_type
            results["pretrain"] = args.pretrain
            results["max_iter"] = args.max_iter

        results["time_elapsed"] = results_no_log.get("time_elapsed")
        all_results.append(results)

    ts = time.strftime('%Y-%m-%d_%H-%M')
    os.makedirs(args.outdir, exist_ok=True)
    filename = os.path.join(args.outdir, f'jax_{ts}_{args.data_type}_{args.n_seeds}_seeds.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
