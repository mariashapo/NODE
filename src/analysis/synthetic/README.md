# Synthetic Analysis Utilities

Quick helpers for inspecting synthetic experiments.

## Plot training data vs predictions
Script: `python -m src.training_convergence.plot_training_fit`

Examples:

- JAX VDP (two-stage pretrain):
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type jax_diffrax \
    --data_type vdp \
    --layer_width "[2,32,2]" \
    --max_iter "[500,10000]" \
    --spacing_type uniform \
    --pretrain "[0.2,1]" \
    --seed 0 \
    --outdir results/plots
  ```

- PyTorch HO:
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type pytorch \
    --data_type ho \
    --max_iter "[400,1000]" \
    --spacing_type uniform \
    --pretrain "[0.2,1]" \
    --seed 1 \
    --outdir results/plots
  ```

- Pyomo HO (Chebyshev spacing by default; adjust tol/reg as needed):
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type pyomo \
    --data_type ho \
    --layer_width "[2,32,2]" \
    --tol 1e-6 \
    --seed 0 \
    --outdir results/plots
  ```

- Multiple models:
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_types pyomo jax_diffrax pytorch \
    --data_type ho \
    --layer_width "[2,32,2]" \
    --max_iter "[400,1000]" \
    --pretrain "[0.2,1]" \
    --plot_mode split_compare \
    --seed 0 \
    --outdir results/plots
  ```
Outputs: PNG under `results/plots/` with train (clean/noisy/pred) and test (clean/pred) for each state.
