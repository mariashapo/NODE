#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_do"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V

# --------------------------
# smoke runs
# --------------------------

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[200,1000]' --pretrain '[0.2,1]' --layer_width "[3,32,2]"\
  --n_seeds 1 --outdir "$OUTDIR" --data_type "do" --penalty_lambda_reg 0.1 --reg_norm --time_invariant False

# --------------------------
# JAX experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[50000]' --pretrain '[1]' --layer_width "[3,32,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "do" --penalty_lambda_reg 0.1 --reg_norm --time_invariant False

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[1000,40000]' --pretrain '[0.2,1]' --layer_width "[3,32,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "do" --penalty_lambda_reg 0.1 --reg_norm --time_invariant False

# --------------------------
# PyTorch experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[1000]' --pretrain '[1]' --layer_width "[3,32,2]" \
  --time_invariant False --n_seeds 20 --outdir "$OUTDIR" --data_type "do"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[400,1000]' --pretrain '[0.2,1]' --layer_width "[3,32,2]" \
  --time_invariant False --n_seeds 20 --outdir "$OUTDIR" --data_type "do"

# --------------------------
# Pyomo regularization experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --no_print True --n_seeds 5 --outdir "results/study_ho_reg" --data_type "ho" --exp "network_size_grid_search"