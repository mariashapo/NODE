#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp_network_size"
DATA_TYPE="ho"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V
micromamba run -n "$ENV" pip install pympler

# --------------------------
# Single Pyomo experiments
micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pyomo \
  --layer_width "[2,32,2]" --penalty_lambda_reg 0.001 --tol 1e-8 --time_invariant True \
  --no_print "False" --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
  --exp "network_size_grid_search"

