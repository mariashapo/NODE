#!/bin/bash
# File: run_pyomo_experiments.sh
# Description: Runs multiple Pyomo and PyTorch convergence experiments

set -e  # exit immediately if any command fails
#set -x  # uncomment if you want to debug / print commands

OUTDIR="results/study01"
mkdir -p "$OUTDIR"

# --------------------------
# Pyomo experiments
# --------------------------
python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,16,2]' --t_range '[0.01,4]' --n_steps 3 \
  --no_print True --n_seeds 1 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,16,2]' --t_range '[0.01,4]' --n_steps 50 \
  --no_print True --n_seeds 20 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pyomo \
  --layer_width '[2,32,2]' --t_range '[0.01,5]' --n_steps 50 \
  --no_print True --n_seeds 20 --outdir "$OUTDIR"

# --------------------------
# PyTorch experiments
# --------------------------
python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[1000]' --pretrain '[1]' \
  --n_seeds 20 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[100,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[200,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[400,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"

python -m src.training_convergnce_studies.training_conv_pytorch \
  --max_iter '[600,1000]' --pretrain '[0.2,1]' \
  --n_seeds 20 --outdir "$OUTDIR"
