import numpy as np
import itertools
import jax.numpy as jnp

import json
import itertools
import sys
import os
import importlib
import logging

def append_path(path):
    if path not in sys.path:
        sys.path.append(path)
        
append_path(os.path.abspath(os.path.join('..', '00_utils')))
append_path(os.path.abspath(os.path.join('..', '00_utils_training')))
append_path(os.path.abspath(os.path.join('..', '00_models')))

logging.basicConfig(level=logging.ERROR, filename='error_log.txt')

import run_train_toy

class ExperimentRunner:
    def __init__(self, config_file):
        # load the config file
        with open(config_file, 'r') as file:
            self.config = json.load(file)
            
        # load the model parameters from the config file
        # params_model will be updated by the update_params_model
        self.params_model = self.config['model_params']
        
        # list to store the tested parameters for 'training_convergence'
        # not required for other optimization types
        self.tested_params = []
            
    def load_trainer(self, data_type, spacing_type="chebyshev", detailed = False):
        """ 
        Load the trainer with the specified data type and spacing type from 'run_train_toy'.
        """
        # reload the module and get the class
        TrainerToy = reload_and_get_attribute(run_train_toy, 'TrainerToy')
        # default data parameters
        data_params_ho = {
            'N': 200,
            'noise_level': self.data_params['noise_level'],
            'ode_type': "harmonic_oscillator",
            'data_param': {"omega_squared": 2},
            'start_time': 0,
            'end_time': 10,
            'spacing_type': spacing_type,
            'initial_state': np.array([0.0, 1.0]),
            'detailed': detailed
        }

        data_params_vdp = {
            'N': 200,
            'noise_level': self.data_params['noise_level'],
            'ode_type': "van_der_pol",
            'data_param': {"mu": 1, "omega": 1},
            'start_time': 0,
            'end_time': 15,
            'spacing_type': spacing_type,
            'initial_state': np.array([0.0, 1.0]),
            'detailed' : detailed
        }

        data_params_do = {
            'N': 200,
            'noise_level': self.data_params['noise_level'],
            'ode_type': "damped_oscillation",
            'data_param': {"damping_factor": 0.1, "omega_squared": 1},
            'start_time': 0,
            'end_time': 10,
            'spacing_type': spacing_type,
            'initial_state': np.array([0.0, 1.0]),
            'detailed' : detailed
        }

        if data_type == "ho":
            p_ = data_params_ho
        elif data_type == "vdp":
            p_ = data_params_vdp
        elif data_type == "do":
            p_ = data_params_do
        else:
            raise ValueError(f"Invalid type {data_type}")

        trainer = TrainerToy(p_, model_type="pyomo")
        trainer.prepare_inputs()
        return trainer
        
    def run(self, optimization_type):
        """
        - Load the trainer with the specified 'data type' and 'spacing type', (self.trainer).
        - Obtain the parameter combinations for the specified optimization type.
        - Loop over the parameter combinations.
            - Update the model parameters. Might also update the trainer, (self.trainer).
            - Train the model with the updated model parameters.
            - Extract the results from the trained model, (results[param_comb]).
        """
        
        # handle 'inf' string in skip_collocation
        if self.params_model['skip_collocation'] == 'inf':
            self.params_model['skip_collocation'] = np.inf
        
        self.data_params = self.config['data']
        self.trainer = self.load_trainer(self.data_params['data_type'], self.data_params['spacing_type'])
        results = {}

        # generate the parameter combinations to loop over for the optimization type
        param_combinations = self.get_param_combinations(optimization_type)
        total_iter = len(param_combinations)
        i = 1

        # loop over the parameter combinations
        for param_comb in param_combinations:
            skip_combination = self.update_params_model(param_comb, optimization_type)
            
            if skip_combination:
                continue
            try:
                self.trainer.train_pyomo(self.params_model)
                if optimization_type == 'training_convergence' and 'optimal' in self.trainer.termination:
                    print(f"Optimal solution found at/before iteration {param_comb}")
                    self.tested_params.append((param_comb[0], param_comb[1]))
            except Exception as e:
                results[param_comb] = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                logging.error(f"Failed to complete training: {e}")
                continue

            try:
                self.extract_results(self.trainer, param_comb, optimization_type, results)
            except Exception as e:
                results[param_comb] = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                logging.error(f"Failed to extract results: {e}")

            print(f"Iteration: {i} / {total_iter}")
            i += 1

        return results, self.trainer

    def get_param_combinations(self, optimization_type):
        """
        Generate the parameter combinations for the specified optimization type.
        - Load the optimization configuration from the config file.
        """
        opt_config = self.config['optimization_types'].get(optimization_type)
        
        if not opt_config:
            raise ValueError(f"Invalid optimization type {optimization_type}")

        if optimization_type == 'regularization':
            param_combinations = opt_config['param_values']

        elif optimization_type == 'tolerances':
            tol_list = opt_config['tol_list']
            param_combinations = list(itertools.product(tol_list, tol_list))

        elif optimization_type == 'reg_tol':
            reg_list = opt_config['reg_list']
            tol_list = opt_config['tol_list']
            param_combinations = list(itertools.product(reg_list, tol_list))

        elif optimization_type == 'skip_collocation':
            param_values = opt_config['param_values']
            # Convert 'inf' string to np.inf
            param_values = [np.inf if v == 'inf' else v for v in param_values]
            param_combinations = param_values

        elif optimization_type == 'training_convergence':
            data = opt_config['data']
            pre_initialize = [opt_config['pre_initialize']]
            l_range = range(opt_config['l_range'][0], opt_config['l_range'][1])
            param_combinations = list(itertools.product(data, pre_initialize, l_range))

        elif optimization_type == 'network_size_grid_search':
            lw_list = opt_config['lw_list']
            reg_list = opt_config['reg_list']
            tol_list = opt_config['tol_list']
            param_combinations = list(itertools.product(lw_list, reg_list, tol_list))

        elif optimization_type == 'activation_function':
            act_func_list = opt_config['act_func_list']
            data = opt_config['data']
            param_combinations = list(itertools.product(act_func_list, data))

        elif optimization_type == 'weights_init':
            weights_init_list = opt_config['weights_init_list']
            data = opt_config['data']
            param_combinations = list(itertools.product(weights_init_list, data))

        elif optimization_type == 'default':
            param_combinations = [None]
        
        else:
            raise ValueError(f"Invalid optimization type {optimization_type}")
        return param_combinations

    def update_params_model(self, param_comb, optimization_type):
        """
        Updates self.params_model with the specified parameter combination for the optimization type.
        """
        skip_combination = False

        if optimization_type == 'regularization':
            self.params_model['penalty_lambda_reg'] = param_comb

        elif optimization_type == 'tolerances':
            tol, constr_tol = param_comb
            self.params_model['params'].update({
                'tol': tol,
                'constr_viol_tol': constr_tol,
                'compl_inf_tol': constr_tol,
                'dual_inf_tol': constr_tol
            })

        elif optimization_type == 'skip_collocation':
            self.params_model['skip_collocation'] = param_comb

        elif optimization_type == 'training_convergence':
            data, pre_init, max_iter = param_comb
            self.params_model['params']['max_iter'] = max_iter

            if max_iter == 1:
                self.trainer = self.load_trainer(data)
                self.params_model['pre_initialize'] = pre_init
                self.tested_params = []

            if (data, pre_init) in self.tested_params:
                skip_combination = True

        elif optimization_type == 'network_size_grid_search':
            lw, reg, tol = param_comb
            self.params_model['layer_widths'] = lw
            self.params_model['penalty_lambda_reg'] = reg
            self.params_model['params']['tol'] = tol

        elif optimization_type == 'activation_function':
            act_func, data = param_comb
            self.params_model['act_func'] = act_func
            self.trainer = self.load_trainer(data)

        elif optimization_type == 'weights_init':
            w_init, data = param_comb
            self.params_model['w_init_method'] = w_init
            self.trainer = self.load_trainer(data)
        
        elif optimization_type == 'default':
            pass    

        return skip_combination

    def extract_results(self, trainer, param_comb, optimization_type, results):
        if optimization_type == 'network_size_grid_search':
            k = (param_comb[0][1], param_comb[1], param_comb[2])
            results[k] = trainer.extract_results_pyomo()
        else:
            results[param_comb] = trainer.extract_results_pyomo()
    
    def extract_solution(self):
        return self.trainer.extract_pyomo_solution()

def reload_and_get_attribute(module, attribute_name):
    """
    Reloads the specified module and retrieves a specified attribute from it.

    Args:
    module: A module object that needs to be reloaded.
    attribute_name: The name of the attribute to retrieve from the module.

    Returns:
    The attribute from the reloaded module.
    """
    reloaded_module = importlib.reload(module)
    return getattr(reloaded_module, attribute_name)