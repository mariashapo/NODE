#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp_net_size"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V


micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,4,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,8,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,16,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,96,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,128,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg 0.01 --time_invariant True --log 250