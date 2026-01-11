#!/usr/bin/env bash
# Run Pytorch training with a specific Pyomo pretrain bundle.

set -euo pipefail

# --- user knobs ---
ENV="node25"                                     # micromamba environment
BUNDLE="results/pyomo_pretrain/pyomo_pretrain_ho_w2-32-2"
OUTDIR="results/pytorch_pretrain_pyomo"
DATA_TYPE="ho"
LAYER_WIDTH="[2,32,2]"
MAX_ITER="[1000]"

# --- run ---
mkdir -p "$OUTDIR" logs

micromamba run -n "$ENV" python -V

micromamba run -n "$ENV" python -m src.synthetic_data.pyomo_pretrain \
  --data_type "$DATA_TYPE" \
  --layer_width "$LAYER_WIDTH" \
  --penalty_lambda_reg 0.1 \
  --tol 1e-8 \
  --spacing_type chebyshev \
  --max_wall_time 5 \
  --seed 42 \
  --outdir results/pyomo_pretrain

micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_pytorch \
  --data_type "$DATA_TYPE" \
  --layer_width "$LAYER_WIDTH" \
  --max_iter "$MAX_ITER" \
  --pretrain "pyomo:$BUNDLE" \
  --penalty_lambda_reg 0.1 \
  --reg_norm "True" \
  --outdir "$OUTDIR" \
  --n_seeds 1

