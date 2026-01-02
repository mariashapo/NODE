# Synthetic Analysis Utilities

Quick helpers for inspecting synthetic experiments.

## Plot training data vs predictions
Script: `python -m src.training_convergence.plot_training_fit`

Examples:

- JAX VDP (two-stage pretrain):
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type jax_diffrax \
    --data_type vdp \
    --layer_width "[2,32,2]" \
    --max_iter "[500,10000]" \
    --spacing_type uniform \
    --pretrain "[0.2,1]" \
    --seed 0 \
    --outdir results/plots
  ```

- PyTorch HO:
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type pytorch \
    --data_type ho \
    --max_iter "[400,1000]" \
    --spacing_type uniform \
    --pretrain "[0.2,1]" \
    --seed 1 \
    --outdir results/plots
  ```

- Pyomo HO (Chebyshev spacing by default; adjust tol/reg as needed):
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_type pyomo \
    --data_type ho \
    --layer_width "[2,32,2]" \
    --spacing_type chebyshev \
    --tol 1e-6 \
    --seed 0 \
    --outdir results/plots
  ```

- Multiple models:
  ```bash
  python -m src.training_convergence.plot_training_fit \
    --model_types pyomo jax_diffrax pytorch \
    --data_type ho \
    --layer_width "[2,32,2]" \
    --pretrain "[0.2,1]" \
    --plot_mode split_compare \
    --seed 0 \
    --outdir results/plots
  ```
Outputs: PNG under `results/plots/` with train (clean/noisy/pred) and test (clean/pred) for each state.

## Aggregate Pyomo regularization sweeps
Example to aggregate and plot a reg curve with CIs (filters tol/layer width and drops combos with <3 runs):
```bash
python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
  --dir results/study_ho_reg/pyomo_ho_241225 \
  --plot \
  --metric mse_test_coll \
  --tol 1e-6 \
  --layer_width "[2,32,2]"
```
Same, but hide the title:
```bash
python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
  --dir results/study_ho_reg/pyomo_ho_241225 \
  --plot \
  --metric mse_test_coll \
  --tol 1e-6 \
  --layer_width "[2,32,2]" \
  --no_title
```

### Quick VDP reg plot (collocation metric, auto tol/width detection)
```bash
python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
  --dir results/study_vdp_reg/pyomo_vdp_251225 \
  --plot \
  --metric mse_test_coll
```
Boxplot variant (per-reg distributions):
```bash
python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
  --dir results/study_vdp_reg/pyomo_vdp_251225 \
  --boxplot \
  --metric mse_test_coll
```


### Hidden layer widths
```bash
python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
  --dir results/study_vdp/pyomo_vdp_layer_width \
  --metric mse_train_coll \
  --plot \
  --x_axis width \
  --reg 0.1 \
  --tol 1e-8 \
  --min_runs 10 --boxplot --no_title
```

Notes:
- The script will load all pickles in the folder. If you omit `--tol`/`--layer_width`, it selects the first tol found and plots each layer width separately.
- If files are corrupt/truncated, they’ll be skipped; ensure the pickles are valid if you see “No records loaded.”
