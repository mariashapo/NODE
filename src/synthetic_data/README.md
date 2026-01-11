# Training Convergence Helpers

## Pyomo Pretraining Bundle
`python -m src.synthetic_data.pyomo_pretrain` trains a Pyomo collocation model and saves weights ready for sequential models.

### What it produces
- `weights_jax`: dict formatted for `nn_jax_diffrax` / TrainerToy JAX.
- `weights_pytorch`: list of (W, b) tuples for `nn_pytorch` / TrainerToy PyTorch.
- `weights_pyomo_raw`: original Pyomo weights.
- Metrics (MSE train/test), timing (solver + wall), termination status, args/model/data metadata.
- Files land in `results/pyomo_pretrain/` by default: a `.pkl` bundle plus a companion `.json` meta.

### Quick VDP example (matches the JAX/PyTorch default width)
```bash
python -m src.synthetic_data.pyomo_pretrain \
  --data_type vdp \
  --layer_width "[2,32,2]" \
  --penalty_lambda_reg 0.01 \
  --tol 1e-12 \
  --spacing_type chebyshev \
  --seed 0
```
After it finishes, use `weights_jax` for JAX runs or `weights_pytorch` for PyTorch runs as the initial parameters when wiring `--pretrain pyomo` in the sequential trainers.

### Notable flags
- `--layer_width`, `--penalty_lambda_reg`, `--tol`, `--act_func`, `--w_init_method`, `--reg_norm`, `--time_invariant`, `--skip_collocation`, `--pre_initialize`, `--redirect_logs`.
- Data alignment: `--spacing_type` (chebyshev/gauss_*), `--noise_level`, `--seed`.
- Outputs: `--outdir` to change the bundle location.

## Using Pyomo pretraining in JAX/PyTorch convergence runs
- New `--pretrain` forms:
  - list of fractions (unchanged): `--pretrain "[0.2,1]"`.
  - Pyomo bundle lookup: `--pretrain pyomo` (auto-picks latest matching bundle in `results/pyomo_pretrain`).
  - Explicit bundle: `--pretrain pyomo:/path/to/bundle.pkl`.
- Optional: `--pyomo_bundle_dir` (defaults to `results/pyomo_pretrain`).
- Timing: the Pyomo pretraining wall time is added to `time_elapsed` and also stored as `pyomo_pretraining_time`; results/meta mark `pyomo_pretraining=True` and record the bundle path.

### Example: JAX with Pyomo pretraining (VDP)
```bash
python -m src.synthetic_data.training_conv_jax \
  --data_type vdp \
  --layer_width "[2,32,2]" \
  --max_iter "[50000]" \
  --pretrain pyomo \
  --outdir results/jax_pretrain_pyomo
```

### Example: PyTorch with Pyomo pretraining (VDP)
```bash
python -m src.synthetic_data.training_conv_pytorch \
  --data_type vdp \
  --layer_width "[2,32,2]" \
  --max_iter "[1000]" \
  --pretrain pyomo \
  --outdir results/pytorch_pretrain_pyomo
```

## Inspecting Pyomo pretrain bundles
Show weights and metrics inside a bundle:
```bash
python -m src.synthetic_data.preview_pyomo_weights results/pyomo_pretrain/pyomo_pretrain_vdp_w2-32-2_seed0_<timestamp>.pkl --suppress
```
