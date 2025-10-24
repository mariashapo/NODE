# run_training.py
import argparse, os, time, pickle
from utils_training.optimize_pyomo_synthetic import ExperimentRunner as PyomoExperimentRunner
from utils.general import generate_seeds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--exp", default="training_convergence_wall_time")
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/pyomo")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runner = PyomoExperimentRunner(args.config)
    all_results = []

    print("STARTING TRAINING")
    for s in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {s}")
        results, trainer = runner.run(args.exp, s)
        all_results.append(results)

    ts = time.strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(args.outdir, f'{ts}_ho_conv_time_{args.n_seeds}_seeds.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
