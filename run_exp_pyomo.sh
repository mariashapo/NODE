#!/usr/bin/env bash
# File: run_experiments.sh

set -euo pipefail

ENV="node25"

# Sanity check (kept, since you asked for it)
micromamba run -n "$ENV" python -V
micromamba run -n "$ENV" pip install pympler

# --------------------------
# Pyomo experiments
# --------------------------
micromamba run -n "$ENV" python -m src.real_life_data.train_pyomo