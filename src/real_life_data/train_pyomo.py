from utils_training.optimize_pyomo_rl import ExperimentRunner
from pathlib import Path
import pickle, json, time

import logging
logging.basicConfig(level=logging.ERROR, filename='error_log.txt')

# repo root is two levels up from src/real_life_data/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "df_train.csv"
OUTDIR = REPO_ROOT / "results" / "pyomo_rl_runs"
OUTDIR.mkdir(parents=True, exist_ok=True)

tol = 1e-6
start_date = '2015-01-15'
extra_input = {}
extra_input['params_data'] = {'file_path': str(DATA_PATH), 'start_date': start_date, 
                'n_points': 300, 'split': 200, 'n_days': 1, 'm': 1, 
                'prev_hour': False, 'prev_week': True, 'prev_year': True, 
                'spacing': 'gauss_radau',
                'encoding': {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'},}

extra_input['params_sequence'] = {'sequence_len': 10, 'frequency': 3}
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

extra_input['plot_collocation'] = False
extra_input['plot_odeint'] = False

runner = ExperimentRunner(start_date, 'converge_wall_time', extra_input)
runner.run()

# Persist results similarly to training_convergence style
ts = time.strftime('%Y-%m-%d_%H-%M-%S')
subdir = OUTDIR / f"pyomo_rl_{ts}"
subdir.mkdir(parents=True, exist_ok=True)

with (subdir / "results_full.pkl").open("wb") as f:
    pickle.dump(runner.results_full, f)

print(f"Saved results to {subdir}")
