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
# PyTorch experiments
# --------------------------
micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[100,100]' --pretrain '[0.2,1]' \
  --n_seeds 1 --outdir "$OUTDIR" --data_type "vdp"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[1000]' --pretrain '[1]' \
  --n_seeds 20 --outdir "$OUTDIR" --data_type "vdp"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[200,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR" --data_type "vdp"

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_pytorch \
  --max_iter '[400,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR" --data_type "vdp"



