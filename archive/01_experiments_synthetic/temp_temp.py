import itertools
import numpy as np
import logging

class ExperimentRunner:
    def run(self, optimization_type):
        tol = 1e-4
        params_model = self.initialize_params_model(tol)
        trainer = self.load_trainer("do")
        results = {}

        param_combinations = self.get_param_combinations(optimization_type)
        total_iter = len(param_combinations)
        i = 1

        for param_comb in param_combinations:
            # Update parameters
            skip_combination = self.update_params_model(params_model, param_comb, optimization_type, trainer)
            if skip_combination:
                continue

            try:
                self.train_model(trainer, params_model)
                # Additional condition for 'training_convergence'
                if optimization_type == 'training_convergence' and 'optimal' in trainer.termination:
                    print(f"Optimal solution found at/before iteration {param_comb}")
                    self.tested_params.append((param_comb[0], param_comb[1]))
            except Exception as e:
                results[param_comb] = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                logging.error("Failed to complete training: {}".format(e))
                print(f"{e}")
                continue

            try:
                self.extract_results(trainer, param_comb, optimization_type, results)
            except Exception as e:
                results[param_comb] = {'time_elapsed': np.nan, 'mse_train': np.nan, 'mse_test': np.nan}
                logging.error("Failed to extract results: {}".format(e))
                print(f"{e}")

            print("Iteration:", i, "/", total_iter)
            i += 1

        return results, trainer

    def initialize_params_model(self, tol):
        params_model = {
            'layer_widths': [2, 32, 2],
            'act_func': 'tanh',
            'penalty_lambda_reg': 0.001,
            'time_invariant': True,
            'w_init_method': 'xavier',
            'reg_norm': False,
            'skip_collocation': np.inf,
            'params': {
                'tol': tol,
                'print_level': 1,
                'max_iter': 3000
            }
        }
        return params_model

    def get_param_combinations(self, optimization_type):
        if optimization_type == 'regularization':
            param_combinations = [1e-6, 0.0001, 0.01, 0.1, 1]

        elif optimization_type == 'tolerances':
            tol_list = [1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
            param_combinations = list(itertools.product(tol_list, tol_list))

        elif optimization_type == 'reg_tol':
            reg_list = [0.001, 0.01, 0.1]
            tol_list = [1e-6, 1e-7, 1e-8]
            param_combinations = list(itertools.product(reg_list, tol_list))

        elif optimization_type == 'skip_collocation':
            param_combinations = [np.inf, 25, 10, 5, 2, 1]

        elif optimization_type == 'training_convergence':
            data = ['ho', 'vdp', 'do']
            pre_initialize = [True]
            l = np.arange(1, 100)  # Must be the last
            param_combinations = list(itertools.product(data, pre_initialize, l))
            self.tested_params = []

        elif optimization_type == 'network_size_grid_search':
            lw_list = [[2, 8, 2], [2, 16, 2], [2, 32, 2], [2, 64, 2], [2, 128, 2]]
            reg_list = [0.0001, 0.001, 0.01]
            tol_list = [1e-4, 1e-6]
            param_combinations = list(itertools.product(lw_list, reg_list, tol_list))

        elif optimization_type == 'activation_function':
            act_func_list = ['tanh', 'softplus', 'sigmoid']
            data = ['ho', 'vdp', 'do']
            param_combinations = list(itertools.product(act_func_list, data))

        elif optimization_type == 'weights_init':
            weights_init_list = ['xavier', 'he', 'random']
            data = ['ho', 'vdp', 'do']
            param_combinations = list(itertools.product(weights_init_list, data))

        elif optimization_type == 'none':
            param_combinations = [0]

        else:
            raise ValueError(f"Invalid optimization type {optimization_type}")

        return param_combinations

    def update_params_model(self, params_model, param_comb, optimization_type, trainer):
        skip_combination = False

        if optimization_type == 'regularization':
            params_model['penalty_lambda_reg'] = param_comb

        elif optimization_type == 'tolerances':
            params_model['params']['tol'] = param_comb[0]
            params_model['params']['constr_viol_tol'] = param_comb[1]
            params_model['params']['compl_inf_tol'] = param_comb[1]
            params_model['params']['dual_inf_tol'] = param_comb[1]

        elif optimization_type == 'skip_collocation':
            params_model['skip_collocation'] = param_comb

        elif optimization_type == 'training_convergence':
            params_model['params']['max_iter'] = param_comb[2]

            if param_comb[2] == 1:
                trainer = self.load_trainer(param_comb[0])
                params_model['pre_initialize'] = param_comb[1]
                self.tested_params = []

            if (param_comb[0], param_comb[1]) in self.tested_params:
                skip_combination = True

        elif optimization_type == 'network_size_grid_search':
            params_model['layer_widths'] = param_comb[0]
            params_model['penalty_lambda_reg'] = param_comb[1]
            params_model['params']['tol'] = param_comb[2]

        elif optimization_type == 'activation_function':
            params_model['act_func'] = param_comb[0]
            trainer = self.load_trainer(param_comb[1])

        elif optimization_type == 'weights_init':
            params_model['w_init_method'] = param_comb[0]
            trainer = self.load_trainer(param_comb[1])

        elif optimization_type == 'none':
            params_model['pre_initialize'] = True

        return skip_combination

    def train_model(self, trainer, params_model):
        trainer.train_pyomo(params_model)

    def extract_results(self, trainer, param_comb, optimization_type, results):
        if optimization_type == 'network_size_grid_search':
            k = (param_comb[0][1], param_comb[1], param_comb[2])
            results[k] = trainer.extract_results_pyomo()
        else:
            results[param_comb] = trainer.extract_results_pyomo()

    def load_trainer(self, data):
        # Placeholder for the actual implementation of load_trainer
        pass
