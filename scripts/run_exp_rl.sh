#!/usr/bin/env bash
# File: run_experiments.sh

set -euo pipefail

ENV="node25"

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V
micromamba run -n "$ENV" pip install pympler

# --------------------------
# experiments
# --------------------------
# pyomo
# micromamba run -n "$ENV" python -m src.real_life_data.train_pyomo --sequence_len 1 --n_seeds 15
# micromamba run -n "$ENV" python -m src.real_life_data.train_pyomo --sequence_len 15 --n_seeds 1

micromamba run -n "$ENV" python -m src.real_life_data.train_jax --n_seeds 1 --sequence_len 15 --exp 'network_size'
micromamba run -n "$ENV" python -m src.real_life_data.train_jax --n_seeds 15 --sequence_len 1 --exp 'network_size'

# quick tests
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 100,10 --pretrain 0.2,1
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 10 --pretrain 1

# actual tests
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 5000,0 --pretrain 0.2,1 --skip-logging-run
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 50000 --pretrain 1

# micromamba run -n "$ENV" python -m src.real_life_data.train_pytorch --num-epochs 100,300 --pretrain 0.2,1
# micromamba run -n "$ENV" python -m src.real_life_data.train_pytorch --num-epochs 500 --pretrain 1
