#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp"
DATA_TYPE="vdp"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V
micromamba run -n "$ENV" pip install pympler

# --------------------------
# Pyomo experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --layer_width '[2,64,2]' --penalty_lambda_reg 0.1 --tol 1e-6 \
  --no_print False --n_seeds 1 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
  --exp "default"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --layer_width '[2,64,2]' --penalty_lambda_reg 0.1 --tol 1e-6 \
  --t_range '[0.01,30]' --n_steps 30 \
  --no_print True --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
  --exp "training_convergence_wall_time"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --layer_width '[2,48,2]' --penalty_lambda_reg 0.075 --tol 1e-6 \
  --t_range '[0.01,25]' --n_steps 25 \
  --no_print True --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
  --exp "training_convergence_wall_time"
