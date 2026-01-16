import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils_training.optimize_pyomo_rl import ExperimentRunner
from datetime import datetime, timedelta
from utils.preprocess import DataPreprocessor
from models.nn_pyomo_admm_1 import NeuralODEPyomoADMM
import pickle
import time
from pathlib import Path


start_date = '2015-01-10'
def generate_dates(start_date, sequence_len = 5, frequency = 2):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    date_sequences = [start_date + timedelta(days=i*frequency) for i in range(sequence_len)]
    date_sequences_str = [date.strftime('%Y-%m-%d') for date in date_sequences]
    return date_sequences_str


# ------------------------------------------------------ SINGLE MODEL RUN ------------------------------------------------------ #
tol = 1e-6
start_date = '2015-01-15'
extra_input = {}
extra_input['params_data'] = {'file_path': '../00_data/df_train.csv', 'start_date': start_date, 
                'n_points': 600, 'split': 360, 'n_days': 1, 'm': 1, 
                'prev_hour': False, 'prev_week': True, 'prev_year': True,
                'spacing': 'gauss_radau',
                'encoding': {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'},}

extra_input['params_sequence'] = {'sequence_len': 15, 'frequency': 5}
extra_input['params_model'] = {'layer_sizes': [7, 32, 1], 'penalty': 1e-7}
extra_input['params_solver'] = { "tol" : tol, 
                         "halt_on_ampl_error" : 'yes', "print_level": 1, "max_iter": 500, 'warm_start_init_point': 'yes'}

extra_input['plot_odeint'] = True

runner = ExperimentRunner(start_date, 'default', extra_input)
runner.run()

single_model_results = runner.results_full
df_single = pd.DataFrame(single_model_results).T
df_single.reset_index(inplace=True)
df_single.rename(columns={'level_0': 'regularization'}, inplace=True)
df_single.drop(columns=['level_1', 'result', 'mse_odeint', 'mse_odeint_test'], inplace=True)
df_single

repo_root = Path(__file__).resolve().parents[2]
out_single = repo_root / "results" / "admm_runs" / "results_nn_pyomo_single.csv"
out_single.parent.mkdir(parents=True, exist_ok=True)
df_single.to_csv(out_single, index=False)

# ------------------------------------------------------ MULTIPLE MODEL RUN ------------------------------------------------------ #
# Constants and Configurations
PENALTY = 1e-5
LAYER_SIZES = [7, 32, 1]
tol = 1e-4
params = {
    "tol": tol,
    "dual_inf_tol": tol,
    "compl_inf_tol": tol,
    "constr_viol_tol": tol,
    "halt_on_ampl_error": 'yes',
    "print_level": 5,
    "max_iter": 500
}
file_path = '../00_data/df_train.csv'
encoding = {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'}

date_sequences_str = generate_dates(start_date, sequence_len = 15, frequency = 5)

def prepare_data(start_date, file_path, encoding):
    data_preprocessor = DataPreprocessor(
        file_path, start_date=start_date, number_of_points = 360, n_days = 1, m = 1, 
        prev_hour = False, prev_week = True, prev_year = True,
        feature_encoding=encoding, split=180, spacing = 'gauss_radau', smooth = False)
    data_subsample = data_preprocessor.load_data()
    return data_preprocessor.preprocess_data(data_subsample), str(data_preprocessor.end_date), data_preprocessor.derivative_matrix()


results = {}
for start_date in date_sequences_str:
    (df_train_1, df_test_1), end_date, (D_1, _) = prepare_data(start_date, file_path, encoding)
    (df_train_2, df_test_2), _, (D_2, D_2_test) = prepare_data(end_date, file_path, encoding)

    # dhift time
    shift = df_train_2['t'].max() - df_train_2['t'].min()
    df_train_2['t'] += shift
    df_test_2['t'] += shift

    # concatenate data
    ys = np.concatenate([df_train_1['y'].values[:, None], df_train_2['y'].values[:, None]], axis = 0)
    ts = np.concatenate([df_train_1['t'], df_train_2['t']])
    Xs = np.concatenate([df_train_1.drop(columns=['y', 't']).values, df_train_2.drop(columns=['y', 't']).values], axis=0)
    Ds = [D_1, D_2]

    # Model and solve
    ode_model = NeuralODEPyomoADMM(
        y_observed=ys, t=ts, first_derivative_matrix=Ds,
        extra_input=Xs, y_init=ys, layer_sizes=LAYER_SIZES,
        act_func="tanh", penalty_lambda_reg=PENALTY, rho=5.0,
        time_invariant=True, w_init_method='xavier', params=params,
        test_data = {'t': df_test_2['t'], 'y': df_test_2['y'], 'D': D_2_test, 'X': df_test_2.drop(columns=['y', 't']).values}
    )
    
    try:
        result = ode_model.admm_solve(iterations = 50, tol_primal=1e-3, record=True)
        results[start_date] = result
    except Exception as e:
        print(f"Error for start date {start_date}: {e}")
        continue

# Save full ADMM results
ts = time.strftime("%Y-%m-%d_%H-%M-%S")
outdir = repo_root / "results" / "admm_runs"
outdir.mkdir(parents=True, exist_ok=True)

# raw dict
with (outdir / f"admm_results_{ts}.pkl").open("wb") as f:
    pickle.dump(results, f)

# optional CSV summary (one row per run)
summary_rows = []
for date_key, res in results.items():
    summary_rows.append({
        "start_date": date_key,
        "mse_coll_train_last": res.get("mse_collocation_train", [None])[-1] if res else None,
        "mse_diffrax_last": res.get("mse_diffrax", [None])[-1] if res else None,
        "mse_test_last": res.get("mse_test_diffrax", [None])[-1] if res else None,
        "mse_coll_test_last": res.get("mse_collocation_test", [None])[-1] if res else None,
        "time_elapsed_last": res.get("time_elapsed", [None])[-1] if res else None,
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(outdir / f"admm_results_summary_{ts}.csv", index=False)
