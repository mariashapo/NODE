#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V

# --------------------------
# JAX experiments
# --------------------------

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[200,1000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]"\
  --n_seeds 1 --outdir "$OUTDIR" --reg_norm

# jax with 64 width layers
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[50000]' --pretrain '[1]' --layer_width "[2,64,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --reg_norm

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --max_iter '[1000,40000]' --pretrain '[0.2,1]' --layer_width "[2,64,2]"\
  --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --reg_norm

