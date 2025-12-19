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
    p.add_argument("--layer_width", type=json.loads, default=None)
    p.add_argument("--reg_norm", action="store_true", default=False)
    p.add_argument("--penalty_lambda_reg", type=float, default=1e-3)
    p.add_argument("--meta", action="store_true", default=True, help="Write run_meta.json with args/params.")
    args = p.parse_args()
    
    params_model = {
        'layer_widths': args.layer_width if args.layer_width is not None else [2, 32, 2],
        'penalty_lambda_reg': args.penalty_lambda_reg,
        'time_invariant': True,
        'learning_rate': 1e-3,
        'max_iter': args.max_iter,
        'pretrain': args.pretrain,
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
    main()
