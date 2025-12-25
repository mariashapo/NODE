#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp_reg"

# Ensure required dirs exist (safe even if you redirect logs outside)
# mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V

# --------------------------
# Pyomo regularization experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --no_print True --n_seeds 15 --outdir "$OUTDIR" --data_type "vdp" --exp "network_size_grid_search"