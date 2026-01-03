#!/usr/bin/env bash
# Run JAX training with a specific Pyomo pretrain bundle.

set -euo pipefail

# --- user knobs ---
ENV="node25"                                     # micromamba environment
BUNDLE="results/pyomo_pretrain/pyomo_pretrain_vdp_w2-64-2_seed42_2025-12-30_15-58-31.pkl"
OUTDIR="results/jax_pretrain_pyomo"
DATA_TYPE="vdp"
LAYER_WIDTH="[2,64,2]"
MAX_ITER="[50000]"

# --- run ---
mkdir -p "$OUTDIR" logs

micromamba run -n "$ENV" python -V

micromamba run -n "$ENV" python -m src.training_convergence.pyomo_pretrain \
  --data_type vdp \
  --layer_width "[2,64,2]" \
  --penalty_lambda_reg 0.1 \
  --tol 1e-8 \
  --spacing_type chebyshev \
  --seed 42 \
  --outdir results/pyomo_pretrain

micromamba run -n "$ENV" python -m src.training_convergence.training_conv_jax \
  --data_type "$DATA_TYPE" \
  --layer_width "$LAYER_WIDTH" \
  --max_iter "$MAX_ITER" \
  --pretrain "pyomo:$BUNDLE" \
  --outdir "$OUTDIR" \
  --n_seeds 15

