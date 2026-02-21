#!/usr/bin/env bash
# Run the ADMM synthetic experiment and store results under results/admm_runs.

set -euo pipefail

ENV="node25"  # adjust to your micromamba env name

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

micromamba run -n "$ENV" python -V

micromamba run -n "$ENV" python -m src.real_life_data.admm

