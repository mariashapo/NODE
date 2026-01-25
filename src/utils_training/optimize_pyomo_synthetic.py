import numpy as np
import itertools
from utils.general import print_memory
import json
import itertools
import importlib
import logging
from pympler import asizeof
import tracemalloc

logging.basicConfig(level=logging.ERROR, filename='error_log.txt')

from utils_training.run_train_toy import TrainerToy

# --- helpers ---
def _hashable_key(param_comb):
    """
    Convert lists (and nested lists/tuples) in param_comb to tuples so it can
    be used as a dict key without altering the original structure elsewhere.
    """
    def _convert(x):
        if isinstance(x, list):
            return tuple(_convert(v) for v in x)
        if isinstance(x, tuple):
            return tuple(_convert(v) for v in x)
        return x
    return _convert(param_comb)


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
        # default data parameters
        data_params_ho = {
            'N': 200,
            'noise_level': self.data_params['noise_level'],
            'ode_type': "harmonic_oscillator",
            'data_param': {"omega_squared": 2},
            'start_time': 0,
            'end_time': 20,
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
    
    def analyse_collocation(self):
        """Analyse the collocation points and derivative matrix used in the current trainer."""
        trainer = self.load_trainer(self.data_type, self.data_params['spacing_type'])
        pass
        
    def run(
        self,
        optimization_type,
        seed=None,
        data_type=None,
        layer_width=None,
        t_range=None,
        n_steps=None,
        penalty_lambda_reg=None,
        tol=None,
    ):
        if self.params_model['skip_collocation'] == 'inf':
            self.params_model['skip_collocation'] = np.inf

        self.data_params = self.config['data']
        self.data_type = data_type if data_type is not None else self.data_params['data_type']
        self.params_model['layer_widths'] = layer_width if layer_width is not None else self.params_model['layer_widths']
        if penalty_lambda_reg is not None:
            self.params_model['penalty_lambda_reg'] = penalty_lambda_reg
        if tol is not None:
            for k in ('tol', 'constr_viol_tol', 'compl_inf_tol', 'dual_inf_tol'):
                self.params_model['params'][k] = tol
        self.results = {}

        param_combinations = self.get_param_combinations(optimization_type, t_range=t_range, n_steps=n_steps)
        print(f"PARAM COMBINATIONS GENERATED: {param_combinations}")
        total_iter = len(param_combinations)
        i = i_since_convergence = 1

        for param_comb in param_combinations:
            skip_combination = self.update_params_model(param_comb, optimization_type, i_since_convergence)
            if skip_combination:
                continue

            trainer = None
            try:
                # fresh trainer per combo
                trainer = self.load_trainer(self.data_type, self.data_params['spacing_type'])
                trainer.train_pyomo(self.params_model, seed)

                if (optimization_type in ['training_convergence', 'training_convergence_wall_time']) \
                and 'optimal' in getattr(trainer, 'termination', ''):
                    print(f"Optimal solution found at/before iteration {param_comb}")
                    self.tested_params.append((param_comb[0], param_comb[1]))
                    i_since_convergence = 1

                # extract -> try to keep only small scalars in results
                try:
                    r = trainer.extract_results_pyomo(detailed = True)
                except Exception as e:
                    r = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                    logging.error(f"Failed to extract results: {e}")


                self.results[_hashable_key(param_comb)] = r

            except Exception as e:
                self.results[_hashable_key(param_comb)] = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                logging.error(f"Failed to complete training: {e}")


            print_memory("Finished training: ")
            # aggressive cleanup
            self._cleanup_trainer(trainer)
            del trainer
            import gc
            gc.collect()
            print_memory("Attempted clean up: ")

            print(f"Iteration: {i} / {total_iter}")
            i_since_convergence += 1
            i += 1

        return self.results, None


    def get_param_combinations(self, optimization_type, t_range = None, n_steps = None):
        """
        Inputs:
            optimisation_type (str) : specify the optimisation_type being executed.
            t_range (None | list) : allow overwriting the time range for *training_convergence_wall_time* optimisation type. e.g. [0.01, 6]
            n_steps (None | int) : allow overwriting the number of steps for *training_convergence_wall_time* optimisation type.
            TODO: [2] the decision for which parameters can be overwritten is currently based on the most-used optimisation types.
                but should be more generalised. 
        
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
            # TODO: [1] this really needs to be cleaned up, but some optimization types allow for multiple data inputs
            # while the rest use the data type specified by the general config
            data = opt_config['data']
            pre_initialize = [opt_config['pre_initialize']]
            l_range = range(opt_config['l_range'][0], opt_config['l_range'][1])
            param_combinations = list(itertools.product(data, pre_initialize, l_range))
            
        elif optimization_type == 'training_convergence_wall_time':
            pre_initialize = [opt_config['pre_initialize']]

            t_start, t_end = t_range if t_range is not None else opt_config['t_range']
            n_steps = n_steps if n_steps is not None else opt_config['n_steps']

            # Nonlinear spacing — more dense near t_start
            exponent = 2.0  # >1 means denser near t_start
            base = np.linspace(0, 1, n_steps)
            wall_times = t_start + (t_end - t_start) * base**exponent
            wall_times = [round(float(t), 5) for t in wall_times]

            # TODO: [1] here we are using data_type from the general config
            param_combinations = list(itertools.product([self.data_type], pre_initialize, wall_times))


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

    def update_params_model(self, param_comb, optimization_type, param_iteration):
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
                self.params_model['pre_initialize'] = pre_init
                self.tested_params = []
            if (data, pre_init) in self.tested_params:
                skip_combination = True

        elif optimization_type == 'training_convergence_wall_time':
            data, pre_init, max_time = param_comb
            self.params_model['params']['max_wall_time'] = max_time
            if param_iteration == 1:
                self.params_model['pre_initialize'] = pre_init
                self.tested_params = []
            if (data, pre_init) in self.tested_params:
                skip_combination = True


        elif optimization_type == 'network_size_grid_search':
            lw, reg, tol = param_comb
            self.params_model['layer_widths'] = lw
            self.params_model['penalty_lambda_reg'] = reg
            # Apply the same tolerance to all key IPOPT tolerance knobs for consistency
            self.params_model['params']['tol'] = tol
            self.params_model['params']['constr_viol_tol'] = tol
            self.params_model['params']['compl_inf_tol'] = tol
            self.params_model['params']['dual_inf_tol'] = tol

        elif optimization_type == 'activation_function':
            act_func, data = param_comb
            self.params_model['act_func'] = act_func
            self.trainer = self.load_trainer(data)

        elif optimization_type == 'weights_init':
            w_init, data = param_comb
            self.params_model['w_init_method'] = w_init
            self.params_model['pre_initialize'] = pre_init
        
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

    def _cleanup_trainer(self, t):
        if t is None:
            return
        # If trainer exposes a close/reset, call it
        try:
            close = getattr(t, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

        for attr in (
            "D", "est_sol", "nodes", "t", "t_test", "true_derivative"
        ):
            if hasattr(t, attr):
                try:
                    setattr(t, attr, None)
                except Exception:
                    pass
    
        t.model.dispose(drop_data=True, drop_params=False, drop_model=True)


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
