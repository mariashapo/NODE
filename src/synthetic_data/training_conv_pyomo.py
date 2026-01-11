# run_training.py
import argparse, os, time, pickle, gc
from utils_training.optimize_pyomo_synthetic import ExperimentRunner as PyomoExperimentRunner
from utils.general import generate_seeds, print_memory, str2bool
import argparse, json
import io, sys, logging, warnings

# --- basic setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pyomo_training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True  # reconfigure even if logging was used before
)

# --- redirect prints (only if later enabled it with --no_print True) ---
class _StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, buf):
        buf = buf.rstrip()
        if buf:
            for line in buf.splitlines():
                self.logger.log(self.level, line)
    def flush(self):
        pass

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--no_print", type=str2bool, default=False)
    p.add_argument("--config", default="src/configs/config_pyomo_synth.json")
    p.add_argument("--exp", default="training_convergence_wall_time") # "default" / "network_size_grid_search" / "training_convergence_wall_time"
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default=None)
    p.add_argument("--data_type", default = "vdp")
    p.add_argument("--layer_width", type=json.loads, default=None)
    p.add_argument("--penalty_lambda_reg", type=float, default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--time_invariant", type=str2bool, default=True)
    p.add_argument("--max_wall_time", type=float, default=None, help="Optional wall-time limit (seconds) for Pyomo solve.")
    # training_convergence_wall_time specific arguments:
    p.add_argument("--t_range", type=json.loads, default=None)
    p.add_argument("--n_steps", type=int, default=1)
    p.add_argument("--meta", action="store_true", default=True, help="Write run_meta.json with args/params.")
    return p

def parse_args(argv=None):
    """Allow passing a custom argv list when debugging/tests; defaults to sys.argv."""
    return _build_parser().parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    if args.no_print:
        print(f"Disabling print statements {args.no_print}.")
        sys.stderr = _StreamToLogger(logging.getLogger("stderr"), logging.ERROR) 
        sys.stdout = _StreamToLogger(logging.getLogger("stdout"), logging.INFO)   # captures print()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    print("STARTING TRAINING")
    run_date = time.strftime('%d%m%y')
    i = 1
    all_results = []
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed} ({i}/{args.n_seeds})")
        print_memory("Memory use loop start: ")
        runner = PyomoExperimentRunner(args.config)
        runner.params_model["time_invariant"] = args.time_invariant
        if args.max_wall_time is not None:
            runner.params_model["params"]["max_wall_time"] = args.max_wall_time
        results, trainer = runner.run(
            args.exp,
            seed=seed,
            data_type=args.data_type,
            layer_width=args.layer_width,
            t_range=args.t_range,
            n_steps=args.n_steps,
            penalty_lambda_reg=args.penalty_lambda_reg,
            tol=args.tol,
        )
        i+=1
        
        print_memory("Training ended: ")
        del trainer, runner
        gc.collect()
        print_memory("Attempted clean up: ")

        ts = time.strftime('%Y-%m-%d_%H-%M')
        # create a dated subfolder for this run
        if args.outdir:
            if args.layer_width is None:
                subdir = os.path.join(args.outdir, f"pyomo_{args.data_type}_{run_date}")
            else:
                subdir = os.path.join(args.outdir, f"pyomo_{args.data_type}_{args.layer_width[1]}_{run_date}")
            os.makedirs(subdir, exist_ok=True)

            if args.meta:
                meta_path = os.path.join(subdir, "run_meta.json")
                if not os.path.exists(meta_path):
                    with open(meta_path, "w") as f:
                        json.dump(
                            {
                                "args": vars(args),
                                "timestamp": ts,
                                "exp": args.exp,
                                "layer_width": args.layer_width,
                                "penalty_lambda_reg": args.penalty_lambda_reg,
                                "tol": args.tol,
                                "t_range": args.t_range,
                                "n_steps": args.n_steps,
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
        else:
            all_results.append((seed, results))
            print(results)
    
    if not args.outdir:
        ts = time.strftime('%Y-%m-%d_%H-%M')
        filename = f"pyomo_all_seeds_{ts}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(all_results, f)
        print(f"All results saved to {filename}")

if __name__ == "__main__":
    action = "prod"
    if action == "dev":
        main([
            "--data_type", "do",
            "--layer_width", "[3,8,2]",
            "--penalty_lambda_reg", "0.1",
            "--tol", "1e-3",
            "--n_seeds", "1",
            "--exp", "default",
            "--time_invariant", "False",
            "--outdir", "results/study_do",
        ])
    else:
        main()
