# Collocation-based Neural ODE Training

## 1. Clone the repository

Clone the repository to your local machine:

```bash
git clone https://github.com/mariashapo/NODE
cd NODE
```

## 2. Set up the environment

Create a new conda environment:

```bash
conda create -n node25_local python=3.9 pip -c conda-forge
conda activate node25_local
python -m pip install -e ".[viz,analysis,utils,pyomo,torch,torch-extras,jax-cpu,optim]"
```

Initialise the base package (if needed):

```bash
pip install -e .
```

## 3. Running the synthetic experiments

Default Pyomo run. Further CLI parameters can be added as needed (refer to `src/synthetic_data/training_conv_pyomo.py` for options).  
Whatever is passed as a CLI argument will override the config file settings.  
The config file for the Pyomo synthetic experiments is located in `src/config_pyomo_synth.json`.

```bash
python -m src.synthetic_data.training_conv_pyomo --layer_width "[2, 32, 2]" --exp default
```

Experiment presets via `--exp`:

- `default`: default experiment (no pre-training).
- `training_convergence_wall_time`: sweeps wall-time limits defined in the config (e.g. `t_range`, `n_steps`) and re-runs multiple Pyomo models to measure convergence vs. wall-clock time (used for convergence visualizations). Optional overrides: `--t_range "[0.01, 10]"` and `--n_steps 30`.
- `network_size_grid_search`: grid over widths / regularization strengths / solver tolerances defined in the config. **Note:** since this script sweeps over parameters, widths/regs/tols in the config override options passed via CLI.

Sample PyTorch runs:

```bash
python -m src.synthetic_data.training_conv_pytorch --max_iter "[1000]" --pretrain "[1]" --layer_width "[2, 32, 2]" --data_type vdp  # without pretraining
python -m src.synthetic_data.training_conv_pytorch --max_iter "[200, 1000]" --pretrain "[0.2, 1]" --layer_width "[2, 32, 2]" --data_type vdp  # with pretraining schedule
```

Note: PyTorch and JAX implementations run each seed twice: one with logging and one without logging, so that the time can be measured without logging overhead in the second run.

## 4. Running real-life data experiments

Download the data from  
<https://drive.google.com/drive/folders/1ehxKYdF-eWPjYj5T6zqrfIXIJNknMZSb?usp=sharing>  
and save it as `data/df_train.csv`.

Default Pyomo run for real-life data experiments. All parameters can be adjusted within `src/real_life_data/train_pyomo.py`; there is no separate config file for this set of experiments.

```bash
python -m src.real_life_data.train_pyomo
```
