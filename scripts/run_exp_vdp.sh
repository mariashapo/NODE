#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp_2"
DATA_TYPE="vdp"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V


micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,25000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]" \
  --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" --penalty_lambda_reg 0.001 --time_invariant True

# Single Pyomo experiments
micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pyomo \
  --time_invariant True --no_print "False" --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
  --exp "training_convergence_wall_time" --layer_width "[2,64,2]" --penalty_lambda_reg 0.001

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pytorch \
  --max_iter '[400,1000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]" \
  --time_invariant True --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" --penalty_lambda_reg 0.001