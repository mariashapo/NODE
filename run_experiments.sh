#!/usr/bin/env bash
# File: run_experiments.sh
# Runs multiple Pyomo and PyTorch convergence experiments via micromamba

set -euo pipefail

ENV="node25"
OUTDIR="results/study01"

# Ensure required dirs exist (safe even if you redirect logs outside)
mkdir -p "$OUTDIR" logs

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V

# --------------------------
# Pyomo experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,16,2]' --t_range '[0.01,4]' --n_steps 3 \
  --no_print True --n_seeds 1 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,16,2]' --t_range '[0.01,4]' --n_steps 50 \
  --no_print True --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,32,2]' --t_range '[0.01,5]' --n_steps 50 \
  --no_print True --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,64,2]' --t_range '[0.01,10]' --n_steps 50 \
  --no_print True --n_seeds 20 --outdir "$OUTDIR"

# --------------------------
# PyTorch experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[1000]' --pretrain '[1]' \
  --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[100,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[200,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[400,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

micromamba run -n "$ENV" python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[600,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"
