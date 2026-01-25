# Collocation-based Neural ODE Training

1. **Clone the Repository**
Clone the repository to your local machine using:
```bash
git clone https://github.com/mariashapo/NODE
cd NODE
```

2. **Set Up the Environment**
Create a new conda environment using the provided `environment.yml` file:
```bash
conda create -n node25_local python=3.9 pip -c conda-forge
conda activate node25_local
python -m pip install -e ".[viz,analysis,utils,pyomo,torch,torch-extras,jax-cpu,optim]"
```

Initialise packages
```bash
pip install -e .
```

3. **Run the Experiments**
```bash
python -m src.synthetic_data.training_conv_pyomo
python -m src.synthetic_data.training_conv_pyomo --layer_width '[2,32,2]' --t_range '[0.01,7]' --n_steps 5 --no_print True
```

## Cross-model Boxplots (Pyomo vs JAX)
- Compare metrics across layer widths:
  ```
  python scripts/compare_boxplots_pyomo_jax.py \
    --pyomo_dir results/study_vdp/pyomo_vdp_32_301225 \
    --jax_dir results/jax_pretrain_pyomo_vdp_w2-64-2/jax_vdp_5000_311225 \
    --metric mse_test \
    --min_runs 3 \
    --out results/plots/pyomo_vs_jax_width.png
  ```
  Uses log-y for MSE; switch to `--metric time_elapsed` for wall-time (linear y).
