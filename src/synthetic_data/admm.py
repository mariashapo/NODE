import pickle
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# jax
import jax.numpy as jnp

from utils_training.run_train_toy import TrainerToy
from models.nn_pyomo_admm import NeuralODEPyomoADMM


def generate_admm_data(initial_state, include_test=False):
    base_params = {
        "N": 150,
        "noise_level": 0.1,
        "ode_type": "van_der_pol",
        "data_param": {"mu": 1, "omega": 1},
        "spacing_type": "gauss_radau",
        "initial_state": initial_state,
        "detailed": False,
    }

    time_intervals = [(0, 15), (15, 30)]

    t_list, y_true_list, y_noisy_list = [], [], []
    D_list = []
    est_sol_list = []
    test_t, test_y, test_D = None, None, None

    for idx, (start_time, end_time) in enumerate(time_intervals):
        params_data = base_params.copy()
        params_data["start_time"] = start_time
        params_data["end_time"] = end_time
        if idx > 0:
            params_data["initial_state"] = y_true_list[-1][-1]

        toy = TrainerToy(params_data, model_type="pyomo")
        toy.prepare_inputs()

        t_list.append(np.array(toy.t))
        y_noisy_list.append(np.array(toy.y_noisy))
        y_true_list.append(np.array(toy.y))
        D_list.append(np.array(toy.D))
        est_sol_list.append(np.array(toy.est_sol))

        if include_test:
            test_t = np.array(toy.t_test)
            test_y = np.array(toy.y_test)
            test_D = np.array(toy.D_test)

    ts = np.concatenate(t_list)
    ys = np.vstack(y_noisy_list)
    y_est = np.hstack(est_sol_list).T

    test_data = None
    if include_test and test_t is not None and test_y is not None:
        test_data = {"t": test_t, "y": test_y, "D": test_D}

    # Optional full-train collocation over entire span
    params_full = base_params.copy()
    params_full["N"] = base_params["N"] * len(time_intervals)
    params_full["start_time"] = time_intervals[0][0]
    params_full["end_time"] = time_intervals[-1][1]
    toy_full = TrainerToy(params_full, model_type="pyomo")
    toy_full.prepare_inputs()
    full_train_data = {
        "t": np.array(toy_full.t),
        "y": np.array(toy_full.y_noisy),
        "D": np.array(toy_full.D),
        "y_est": np.array(toy_full.est_sol),
    }

    return {"ts": ts, "ys": ys, "y_est": y_est, "Ds": D_list, "test_data": test_data, "full_train": full_train_data}


def main():
    seeds = [random.randint(0, 10_000_000) for _ in range(30)]
    use_full_train = False
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
        data = generate_admm_data(np.array([0.0, 1.0]), include_test=True)
        ts = data["ts"]
        ys = data["ys"]
        y_est = data["y_est"]
        Ds = data["Ds"]
        test_data = data["test_data"]
        full_train = data["full_train"]

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
            full_train=full_train,
            use_full_train=use_full_train,
        )

        res = ode_model.admm_solve(iterations=20, tol_primal=1e-2, record=True)
        res["seed"] = seed
        results.append(res)

    # … after the for-loop finishes
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(__file__).resolve().parents[2] / "results" / "admm_runs"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"admm_results_{ts}.pkl").open("wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved ADMM results to {outdir}")


if __name__ == "__main__":
    main()
