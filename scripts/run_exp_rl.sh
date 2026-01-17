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
# quick tests
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 100,10 --pretrain 0.2,1
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 10 --pretrain 1

# actual tests
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 5000,0 --pretrain 0.2,1 --skip-logging-run
# micromamba run -n "$ENV" python -m src.real_life_data.train_jax --num-epochs 50000 --pretrain 1

# micromamba run -n "$ENV" python -m src.real_life_data.train_pytorch --num-epochs 100,300 --pretrain 0.2,1
# micromamba run -n "$ENV" python -m src.real_life_data.train_pytorch --num-epochs 500 --pretrain 1

micromamba run -n "$ENV" python -m src.real_life_data.train_pytorch \
  --num-epochs 300 --pretrain 1 \
  --dates "2015-01-15,2015-01-18,2015-01-21,2015-01-27,2015-01-30,2015-02-05,2015-02-11,2015-02-14,2015-02-17,2015-02-26,2015-03-01,2015-03-04,2015-03-07,2015-03-13,2015-03-25,2015-03-28,2015-03-31,2015-04-03,2015-04-06,2015-04-12"
