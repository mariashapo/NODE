#!/usr/bin/env bash
# Runs multiple Pyomo/PyTorch convergence experiments on cluster via micromamba env.

set -euo pipefail

ENV_NAME="node25"
OUTDIR="results/study_vdp_net_size"
DATA_TYPE="vdp"
WIDTHS=(4 8 16 32 64 128)

mkdir -p "$OUTDIR" logs

micromamba run -n "$ENV_NAME" python -V

for w in "${WIDTHS[@]}"; do
  # Pyomo: time_invariant False, A=1
  micromamba run -n "$ENV_NAME" python -m src.synthetic_data.training_conv_pyomo \
    --time_invariant False --no_print False --n_seeds 10 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
    --exp training_convergence_wall_time --t_range "[5,100]" --n_steps 5 --layer_width "[3,$w,2]" --penalty_lambda_reg 0.001 --A 1

  # Pyomo: time_invariant True, A=0
  micromamba run -n "$ENV_NAME" python -m src.synthetic_data.training_conv_pyomo \
    --time_invariant True --no_print False --n_seeds 10 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
    --exp default --layer_width "[2,$w,2]" --penalty_lambda_reg 0.001 --A 0

  # PyTorch: time_invariant False, A=1
  micromamba run -n "$ENV_NAME" python -m src.synthetic_data.training_conv_pytorch \
    --max_iter "[400,1000]" --pretrain "[0.2,1]" --layer_width "[3,$w,2]" \
    --time_invariant False --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
    --penalty_lambda_reg 0.001 --A 1 --log 200

  # PyTorch: time_invariant True, A=0
  micromamba run -n "$ENV_NAME" python -m src.synthetic_data.training_conv_pytorch \
    --max_iter "[400,800]" --pretrain "[0.2,1]" --layer_width "[2,$w,2]" \
    --time_invariant True --n_seeds 15 --outdir "$OUTDIR" --data_type "$DATA_TYPE" \
    --penalty_lambda_reg 0.001 --A 0 --log 200
done
