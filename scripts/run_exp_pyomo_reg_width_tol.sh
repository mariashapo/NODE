#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_ho_reg"
DATA_TYPE="ho"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V
micromamba run -n "$ENV" pip install pympler

# --------------------------
# Pyomo experiments
# --------------------------

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pyomo \
  --no_print True --n_seeds 1 --outdir "$OUTDIR" --data_type "$DATA_TYPE" --exp "default"

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pyomo \
  --no_print True --n_seeds 5 --outdir "$OUTDIR" --data_type "$DATA_TYPE" --exp "network_size_grid_search"


