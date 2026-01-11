#!/usr/bin/env bash
# JAX/Diffrax width sweep using staged pretraining fractions ([0.2,1]).

set -euo pipefail

ENV="node25"
OUTDIR="results/study_vdp/jax_vdp_layer_width"
DATA_TYPE="vdp"
MAX_ITER="[1000,20000]"
PRETRAIN='[0.2,1]'  # staged pretraining fractions
N_SEEDS=5
TIMING_ONLY=True
WIDTHS=(
  "[2,4,2]"
  "[2,8,2]"
  "[2,16,2]"
  "[2,32,2]"
  "[2,64,2]"
  "[2,96,2]"
  "[2,128,2]"
)

mkdir -p "$OUTDIR" logs
micromamba run -n "$ENV" python -V

for W in "${WIDTHS[@]}"; do
  micromamba run -n "$ENV" python -m src.synthetic_data.training_conv_jax \
    --data_type "$DATA_TYPE" \
    --layer_width "$W" \
    --max_iter "$MAX_ITER" \
    --pretrain "$PRETRAIN" \
    --outdir "$OUTDIR" \
    --n_seeds "$N_SEEDS" \
    --timing_only "$TIMING_ONLY"
done
