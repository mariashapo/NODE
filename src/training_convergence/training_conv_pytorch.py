"""There no dedicated experiment runner for the synthetic Pytorch model, the same way there is one for Pyomo (PyomoExperimentRunner)."""
import argparse, os, time, pickle
from utils.general import generate_seeds
import argparse, json
from utils_training.run_train_toy import TrainerToy as Trainer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--outdir", default="results/pytorch")
    p.add_argument("--data_type", default = "ho")
    p.add_argument("--max_iter", type=json.loads, default=[400, 1000])
    p.add_argument("--pretrain", type=json.loads, default=[0.2, 1])
    args = p.parse_args()
    
    params_model = {
        'layer_widths': [2, 32, 2],
        'penalty_lambda_reg': 1e-3,
        'time_invariant': True,
        'learning_rate': 1e-3,
        'max_iter': args.max_iter,
        'pretrain': args.pretrain,
        'split_time': True,
        'rtol': 1e-3,
        'atol': 1e-6,
    }
    
    
    all_results = []
    print("STARTING TRAINING")
    for seed in generate_seeds(args.n_seeds):
        print(f"EXECUTING SEED {seed}")
        trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type = "pytorch")
        params_model["log"] = True
        trainer.train(params_model, seed = seed)
        results = trainer.extract_results_pytorch()
        results['train_loss'] = trainer.losses
        results['data_type'] = args.data_type
        results['pretrain'] = args.pretrain
        results['max_iter'] = args.max_iter
        # time should be measured off the model with no exta logging computations!!!
        trainer = Trainer.load_trainer(args.data_type, spacing_type="uniform", model_type = "pytorch")
        params_model["log"] = False
        trainer.train(params_model, seed = seed)
        results_no_log = trainer.extract_results_pytorch()
        results["time_elapsed"] = results_no_log["time_elapsed"]
        all_results.append(results)

    ts = time.strftime('%Y-%m-%d_%H-%M')
    os.makedirs(args.outdir, exist_ok=True)
    filename = os.path.join(args.outdir, f'pytoch_{ts}_{args.data_type}_{args.n_seeds}_seeds.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
