import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# jax
import jax.numpy as jnp

from utils.data_generation import DataPreprocessor
# from utils.non_parametric_collocation import collocate_data
# from utils.collocation_obj import Collocation
from models.nn_pyomo_admm import NeuralODEPyomoADMM


def generate_admm_data(initial_state, include_test=False):
    data_params_vdp = {
        'N': 150, 'noise_level': 0.1,
        'ode_type': "van_der_pol",
        'extra_param': {"mu": 1, "omega": 1},
        'spacing_type': 'gauss_radau',
        'initial_state': initial_state,
        'detailed' : False
    }

    time_intervals = [(0, 15), (15, 30)]

    t_list, y_true_list, y_noisy_list = [], [], []
    D_list = []
    est_sol_list = []

    # --------------------------------------------- GENERATE DATA FOR EACH INTERVAL --------------------------------------------- #
    test_t, test_y = None, None
    for idx, (start_time, end_time) in enumerate(time_intervals):
        # update initial state for subsequent intervals
        if idx > 0:
            data_params_vdp['initial_state'] = y_true_list[-1][-1] 
        
        data_params_vdp['start_time'] = start_time
        data_params_vdp['end_time'] = end_time
        
        data_prep = DataPreprocessor(data_params_vdp)
        data_prep.load_data()
        data_prep.prepare_collocation()
        data_prep.estimate_derivative()
        test_t, test_y = getattr(data_prep, "t_test", None), getattr(data_prep, "y_test", None)
        
        # training
        y_noisy_list.append(data_prep.y_noisy)
        t_list.append(data_prep.t)
        D_list.append(data_prep.D)
        y_true_list.append(data_prep.y)
        est_sol_list.append(data_prep.est_sol)
        
    # --------------------------------------------- MERGE DATA FROM ALL INTERVALS --------------------------------------------- #
    ts = np.concatenate(t_list)      
    ys = np.vstack(y_noisy_list)     
    y_est = np.hstack(est_sol_list)  
    Ds = D_list          

    ys = np.array(ys)
    ts = np.array(ts)
    y_est = np.array(y_est).T
    
    if include_test:
        test_data = None
        if test_t is not None and test_y is not None:
            test_data = {"t": np.array(test_t), "y": np.array(test_y)}
        return ts, ys, y_est, Ds, test_data
    
    return ts, ys, y_est, Ds


def main():
    seeds = [random.randint(0, 10_000_000) for _ in range(30)]
    tol = 1e-8
    params = {
        "tol": tol,
        "halt_on_ampl_error": "yes",
        "print_level": 5,
        "max_iter": 3000,
    }
    layer_sizes = [2, 32, 2]
    results = []

    for seed in seeds:
        ts, ys, y_est, Ds, test_data = generate_admm_data(np.array([0.0, 1.0]), include_test=True)

        ode_model = NeuralODEPyomoADMM(
            y_observed=ys,
            t=ts,  # t
            first_derivative_matrix=Ds,  # derivative matrix
            extra_input=None,  # extra inputs
            y_init=y_est,
            layer_sizes=layer_sizes,
            act_func="tanh",
            penalty_lambda_reg=0.001,
            rho=1.0,
            time_invariant=True,
            w_init_method="xavier",
            params=params,
            test_data=test_data,
            seed=seed,
        )

        results.append(ode_model.admm_solve(iterations=20, tol_primal=1e-2, record=True))

    bp = 1


if __name__ == "__main__":
    main()
