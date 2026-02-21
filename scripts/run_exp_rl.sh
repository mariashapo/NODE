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
# micromamba run -n "$ENV" python -m src.real_life_data.train_pyomo --sequence_len 1 --n_seeds 15 --exp 'network_size'

# jax
micromamba run -n "$ENV" python -m src.real_life_data.train_pyomo --n_seeds 1 --sequence_len 15 --sequence_freq 5 --exp 'converge_wall_time'
