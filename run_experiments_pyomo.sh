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
  --layer_width '[2,8,2]' --t_range '[0.01,5]' --n_steps 2 \
  --no_print True --n_seeds 1 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --layer_width '[2,32,2]' --t_range '[0.01,20]' --n_steps 30 \
  --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
  --layer_width '[2,32,2]' --t_range '[0.01,20]' --n_steps 30 \
  --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

# micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
#   --layer_width '[2,16,2]' --t_range '[0.01,10]' --n_steps 30 \
#   --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

# micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
#   --layer_width '[2,16,2]' --t_range '[0.01,10]' --n_steps 30 \
#   --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

# micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
#   --layer_width '[2,64,2]' --t_range '[0.01,15]' --n_steps 30 \
#   --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

# micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
#   --layer_width '[2,8,2]' --t_range '[0.01,15]' --n_steps 30 \
#   --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

# micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pyomo \
#   --layer_width '[2,16,26,2]' --t_range '[0.01,15]' --n_steps 30 \
#   --no_print True --n_seeds 10 --outdir "$OUTDIR" --data_type"$DATA_TYPE"

