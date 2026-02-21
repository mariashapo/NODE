#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple JAX convergence experiments via micromamba (lambda sweep)

set -euo pipefail

ENV="node25"
OUTDIR="results/jax_study_reg_ho"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V

for LAMBDA in 0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1; do
  micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
    --max_iter '[1000,5000]' --pretrain '[0.2,1]' --layer_width "[2,32,2]" \
    --n_seeds 10 --outdir "$OUTDIR" --data_type "ho" --penalty_lambda_reg "$LAMBDA" \
    --time_invariant True --log 500
done


OUTDIR="results/jax_study_reg_vdp"

for LAMBDA in 0 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1; do
  micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
    --max_iter '[1000,5000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]" \
    --n_seeds 10 --outdir "$OUTDIR" --data_type "vdp" --penalty_lambda_reg "$LAMBDA" \
    --time_invariant True --log 500
done