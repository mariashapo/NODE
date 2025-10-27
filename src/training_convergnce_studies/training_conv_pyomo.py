# run_training.py
import argparse, os, time, pickle
from utils_training.optimize_pyomo_synthetic import ExperimentRunner as PyomoExperimentRunner
from utils.general import generate_seeds
import argparse, json
import io, sys, logging, warnings

# 2) Send Python warnings into logging (instead of terminal)
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=UserWarning)

# # 3) Redirect *print()* and uncaught errors to logging (no terminal output)
class _StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, buf):
        buf = buf.rstrip()
        if buf:
            for line in buf.splitlines():
                self.logger.log(self.level, line)
    def flush(self):  # needed for some libraries
        pass

stdout_logger = logging.getLogger("stdout")
stderr_logger = logging.getLogger("stderr")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no_print", type=bool, default = False)
    p.add_argument("--config", default="src/configs/config_pyomo_synth.json")
    p.add_argument("--exp", default="training_convergence_wall_time")
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/pyomo")
    p.add_argument("--data_type", default = "ho")
    p.add_argument("--layer_width", type=json.loads, default=None)
    # training_convergence_wall_time specific arguments:
    p.add_argument("--t_range", type=json.loads, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    args = p.parse_args()

    if args.no_print:
        print(f"Disabling print statements {args.no_print}.")
        sys.stderr = _StreamToLogger(stderr_logger, logging.ERROR) 
        sys.stdout = _StreamToLogger(stdout_logger, logging.INFO)   # captures print()

    os.makedirs(args.outdir, exist_ok=True)
    runner = PyomoExperimentRunner(args.config)
    all_results = []

    print("STARTING TRAINING")
    i = 1
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed} ({i}/{args.n_seeds})")
        # run(self, optimization_type, seed = None, data_type = None, layer_width = None, t_range = None, n_steps = None)
        results, _ = runner.run(args.exp, seed = seed, data_type = args.data_type, layer_width = args.layer_width, t_range = args.t_range, n_steps = args.n_steps)
        all_results.append(results)
        i+=1

    ts = time.strftime('%Y-%m-%d_%H-%M')
    filename = os.path.join(args.outdir, f'{ts}_{args.data_type}_conv_time_{args.n_seeds}_seeds.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
