# Collocation-based Neural ODE Training

This project develops and tests a collocation-based methodology for efficiently training Neural Ordinary Differential Equations (Neural ODEs). 
By leveraging spectral methods and incorporating interpolation techniques, this framework aims to enhance the speed and accuracy of Neural ODE training processes. 

## Key Features

- **Collocation-Based Training:** Utilizes collocation methods to integrate constraints directly within the training process, enhancing the stability and convergence speed.
- **Hybrid Training Framework:** Combines traditional and collocation-based techniques to optimize model training and performance, especially on complex tasks.
- **Spectral Methods Integration:** Explores the use of spectral methods beyond traditional collocation approaches, potentially offering more robust and scalable solutions.
- **Real-World Application:** Tests the methodology on real-life data, assessing its effectiveness in practical scenarios.

## Key Files
```bash
NODE/
│
├── README.md          
├── environment.yml                     # The environment file for Python projects
├── models/         
│   ├── nn_jax_diffrax.py               # Diffrax-JAX benchmark model
│   ├── nn_pytorch.py                   # Pytorch benchmark model
│   ├── nn_pyomo_base.py                # Collocation-based Pyomo model
│   ├── nn_pyomo_admm.py                # Collocation-based ADMM Pyomo model
│   └── ode_solver_pyomo_base.py        # Direct Collocation ODE solver
│   
├── utils/            
│   ├── collocation_obj.py              # Collocation: Differential matrix, grid computation
│   ├── data_genration.py               # Data generation for synthetic data
│   ├── preprocessing.py                # Data preprocessing for real-world data
│   ├── non_parametric_collocation.py   # Least squares approximation for smoothing
│   └── analyse_results.py              # Helper function for analysis of results
│
├── utils_training/                  # Files to assist training and collecting results
│   ├── optimize_diffrax_rl.py          # Hyperparam. optim. for Diffrax model (real data)
│   ├── optimize_pyomo_rl.py            # Hyperparam. optim. for Pytorch model (real data)
│   ├── run_train_diffrax_rl.py         # Training script for Diffrax model (real data)
│   ├── run_train_pyomo_rl.py           # Training script for Pytorch model (real data)
│   ├── utils_pytorch.py                # Training script for Pytorch model (real data)
│   └── run_train_toy.py                # Training script for All models (synthetic data)
│
├── 01_experiments_synthetic/           # Experiemnts on synthetic data
│   ├── analysis_notebooks/             # Notebooks used to analyze results
│   ├── 00_training_toy_diffrax.ipynb   # Training entry point for Diffrax model
│   ├── 00_training_toy_pyomo.ipynb     # Training entry point for Pyomo model
│   └── 00_training_toy_pytorch.ipynb   # Training entry point for Pytorch model
│
├── 02_experiments_real_life/           # Experiments on real data
│   ├── analysis_notebooks/             # Notebooks used to analyze results
│   ├── 00_train_diffrax.ipynb          # Training entry point for Diffrax model
│   ├── 00_train_pyomo.ipynb            # Training entry point for Pyomo model
│   └── 00_train_pytorch.ipynb          # Training entry point for Pytorch model
│
├── 03_admm/                            # ADMM-based collocation
│   └── train_pyomo_admm_rl.ipynb       # Training entry point for ADMM Pyomo model
.
```

## Running the Code

1. **Clone the Repository**
Clone the repository to your local machine using:
```bash
git clone https://github.com/mariashapo/NODE
cd NODE
```

2. **Set Up the Environment**
Create a new conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml node
conda activate node
```

Initialise packages
```bash
pip install -e .
```

3. **Run the Experiments**
Navigate to the desired experiment folder (e.g., `02_experiments_real_life`) and run the entry point notebooks `00_train_diffrax.ipynb`, `00_train_pyomo.ipynb`, or `00_train_pytorch.ipynb`.

Sample parameters for the Pyomo model:
```python
tol = 1e-6
start_date = '2015-01-15'
extra_input = {}
extra_input['params_data'] = {'file_path': '../00_data/df_train.csv', 'start_date': start_date, 
                'n_points': 300, 'split': 200, 'n_days': 1, 'm': 1, 
                'prev_hour': False, 'prev_week': True, 'prev_year': True, 
                'spacing': 'gauss_radau',
                'encoding': {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'},}

extra_input['params_sequence'] = {'sequence_len': 1, 'frequency': 35}
extra_input['params_model'] = {'layer_sizes': [7, 32, 1], 'penalty': 1e-5, 'w_init_method': 'xavier'}
extra_input['params_solver'] = { 
                        "tol":tol, 
                        "dual_inf_tol": 0.1, 
                        "compl_inf_tol": tol,
                        "constr_viol_tol": 1e-8, 
                        'warm_start_init_point': 'yes',
                        "halt_on_ampl_error" : 'yes',
                        "print_level": 5, "max_iter": 3000,
                        "bound_relax_factor": 1e-8
                        }

extra_input['plot_collocation'] = True
extra_input['plot_odeint'] = True
```

```bash
conda activate node_25
python -m pip install -e .
python -m src.synthetic_data.training_conv_pyomo
python -m src.synthetic_data.training_conv_pyomo --layer_width '[2,32,2]' --t_range '[0.01,7]' --n_steps 5 --no_print True
```

## Quick commands (training convergence sweeps)
- Pyomo batch (VDP): `bash run_experiments_pyomo.sh` or `bash run_exp_vdp.sh`
- Regularization/width/tol grid (Pyomo VDP): `bash run_exp_pyomo_reg_width_tol.sh`
- PyTorch batches (VDP): `bash run_experiments_pytorch.sh`
- JAX/Diffrax batches (HO/VDP): `bash run_experiments_jax.sh`
- Inspect results: notebooks under `src/analysis/synthetic/` (e.g., `00_training_convergence.ipynb`); aggregate Pyomo reg search via `python src/analysis/synthetic/aggregate_pyomo_reg_search.py`

## Regularization Study (Pyomo, VDP)
- Launch sweep (reg/width/tol grid): `bash run_exp_pyomo_reg_width_tol.sh`
- Aggregate/plot (reg on x-axis by default):
  ```
  python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
    --dir results/study_vdp_reg/pyomo_vdp \
    --metric mse_test \
    --plot \
    --reg 0.1 --tol 1e-8 \
    --x_axis reg
  ```
  Add `--boxplot` for boxplots; adjust `--metric`/`--reg`/`--tol` as needed.

## Layer Width Study (Pyomo, VDP)
- Launch sweep (width/reg/tol grid): `bash run_exp_pyomo_reg_width_tol.sh` (uses config lists for widths/regs/tols).
- Aggregate/plot width on x-axis (works for `results/study_vdp/pyomo_vdp_*`):
  ```
  python -m src.analysis.synthetic.aggregate_pyomo_reg_search \
    --dir results/study_vdp/pyomo_vdp_32_301225 \
    --metric mse_test \
    --plot \
    --x_axis width \
    --reg 0.1 \
    --tol 1e-8 \
    --min_runs 3
  ```
  Swap `--metric time_elapsed` to plot wall time; use `--boxplot` for boxplots; `--min_runs` controls the minimum seeds per point.

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
