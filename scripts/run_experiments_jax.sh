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
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0 --time_invariant True --log 500

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.00000001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.0000001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.000001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.00001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,96,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.0001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,128,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.001 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,128,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.01 --time_invariant True --log 250

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,128,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 0.1 --time_invariant True --log 250

  micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
  --max_iter '[1000,10000]' --pretrain '[0.2,1]' --layer_width "[2,128,2]"\
  --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg 1 --time_invariant True --log 250