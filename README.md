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
├── 00_models/         
│   ├── nn_jax_diffrax.py               # Diffrax-JAX benchmark model
│   ├── nn_pytorch.py                   # Pytorch benchmark model
│   ├── nn_pyomo_base.py                # Collocation-based Pyomo model
│   ├── nn_pyomo_admm.py                # Collocation-based ADMM Pyomo model
│   └── ode_solver_pyomo_base.py        # Direct Collocation ODE solver
│   
├── 00_utils/            
│   ├── collocation_obj.py              # Collocation: Differential matrix, grid computation
│   ├── data_genration.py               # Data generation for synthetic data
│   ├── preprocessing.py                # Data preprocessing for real-world data
│   ├── non_parametric_collocation.py   # Least squares approximation for smoothing
│   └── analyse_results.py              # Helper function for analysis of results
│
├── 00_utils_training/                  # Files to assist training and collecting results
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