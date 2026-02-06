# Collocation-based Neural ODE Training

# 1. Clone the repository

Clone the repository to your local machine:

```bash
git clone https://github.com/mariashapo/NODE
cd NODE
```

# 2. Set up the environment

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

# 3. Running the synthetic experiments

Default Pyomo run. Further CLI parameters can be added as needed (refer to `src/synthetic_data/training_conv_pyomo.py` for options).  
Whatever is passed as a CLI argument will override the config file settings.  
The config file for the Pyomo synthetic experiments is located in `src/config_pyomo_synth.json`.

```bash
python -m src.synthetic_data.training_conv_pyomo --layer_width "[2, 32, 2]" --exp default
```

Experiment presets via `--exp`:

- `default`: default experiment (single run).
- `training_convergence_wall_time`: sweeps wall-time limits defined in the config (e.g. `t_range`, `n_steps`) and re-runs multiple Pyomo models to measure convergence vs. wall-clock time (used for convergence visualizations). Optional overrides: `--t_range "[0.01, 10]"` and `--n_steps 30`.
- `network_size_grid_search`: grid over widths / regularization strengths / solver tolerances defined in the config. **Note:** since this script sweeps over parameters, widths/regs/tols in the config override options passed via CLI.

Sample PyTorch runs:

```bash
python -m src.synthetic_data.training_conv_pytorch --max_iter "[1000]" --pretrain "[1]" --layer_width "[2, 32, 2]" --data_type vdp  # without pretraining
python -m src.synthetic_data.training_conv_pytorch --max_iter "[200, 1000]" --pretrain "[0.2, 1]" --layer_width "[2, 32, 2]" --data_type vdp  # with pretraining schedule
```

Note: PyTorch and JAX implementations run each seed twice: one with logging and one without logging, so that the time can be measured without logging overhead in the second run.


# 4. Running real-life data experiments

Download the data from  
<https://drive.google.com/drive/folders/1ehxKYdF-eWPjYj5T6zqrfIXIJNknMZSb?usp=sharing>  
and save it as `data/df_train.csv`.

Default Pyomo run for real-life data experiments. All parameters can be adjusted within `src/real_life_data/train_pyomo.py`; there is no separate config file for this set of experiments.

Basic parameters can be passed via CLI:
```bash
python -m src.real_life_data.train_pyomo --n_seeds 1 --sequence_len 1 
```
- `n_seeds`: number of random seeds to run and `sequence_len`: length of the date sequences to run.

Experiment presets can be accssed ExperimentRunner using the `--exp` argument, similarly to the in synthetic experiments. Optimization presets (corresponding to `--exp`) can be modified in the define_param_combinations() method of the ExperimentRunner class. Sample preset:
- `default`: default experiment (single run for each date and seed).
- `network size`: grid over widths / regularization strengths / solver tolerances defined in the config.

Sample JAX run for real-life data experiments:
```bash
python -m src.real_life_data.train_jax --n_seeds 1 --sequence_len 1 --exp 'default'
```

Example of an experiment with multiple seeds and sequence lengths:
```bash
python -m src.real_life_data.train_jax --n_seeds 1 --sequence_len 1 --exp 'network_size'
```

# 5. Visualization and analysis

## 5.1 Synthetic experiments

#### Predicted Trajectories:
To plot the predicted trajectories after training, `plot_training_fit.py` can be used. Sample command (adjust parameters as needed, refer to `src/synthetic_data/plot_training_fit.py` for options):
```bash
python -m src.synthetic_data.plot_training_fit --model_type pyomo --data_type ho --layer_width "[2,32,2]" --penalty_lambda_reg 0.1 --tol 1e-8 --seed 42 --outdir results/plots/traj
```

Optional argument `--x_ticks_endpoints` can be used to only include the key timepoints on the x-axis (start, train/test split, end).

#### Convergence plots:
To study the convergence behaviour, the following notebooks are used: `src/analysis/aynthetic/training_convergence_{ho/vdp/do}.ipynb` for each system. These noteboooks rely on having pickeled results in the `syntehtic/results` folder. 
To generate the pickled results, run the `training_convergence_wall_time` experiment preset as described in section 3 and place the results in the `synthetic/results` folder.

#### Regularization analysis:

Sample script to plot the regularization study results (the directory should point to the results of a `network_size_grid_search` experiment):
```bash
python -m src.analysis.synthetic.pyomo_reg_search_plot \
  --dir src/analysis/synthetic/results/study_ho_reg/pyomo_ho_241225 \
  --plot \
  --metric mse_test_coll \
  --tol 1e-6 \
  --layer_width "[2,32,2]" \
  --save --no_title
```

#### Pareto front analysis:
```bash
python -m src.analysis.synthetic.pyomo_reg_search_plot \
  --dir src/analysis/synthetic/results/study_ho_reg/pyomo_ho_241225 \
  --metric mse_train_coll \
  --tol 1e-6 \
  --layer_width "[2,32,2]" \
  --save --no_title --plot
  ```


## 5.2 Real-life data experiments
#### Predicted Trajectories:

Predicted trajectory plots are generated within the training scripts after training is completed.

Sample command for Pyomo predicted trajectory plotting is the same as the default run (make sure that either `extra_input['plot_collocation']` or `extra_input['plot_odeint']` is set to `True` in `train_pyomo.py`):
```bash
python -m src.real_life_data.train_pyomo --sequence_len 1 --n_seeds 1
```

#### Network size analysis:

Sample commands to loop through different network sizes:
```bash
micromamba run -n "$ENV" python -m src.real_life_data.train_jax --n_seeds 1 --sequence_len 15 --exp 'network_size' # option with 15 different dates
micromamba run -n "$ENV" python -m src.real_life_data.train_jax --n_seeds 15 --sequence_len 1 --exp 'network_size' # option with 15 different seeds for the same date
```