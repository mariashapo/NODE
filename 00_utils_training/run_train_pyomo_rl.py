import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import sys
import os
import shutil
import time
import pickle
import importlib

def append_path(path):
    if path not in sys.path:
        sys.path.append(path)

def reload_module(module_name, class_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return getattr(module, class_name)
        
append_path(os.path.abspath(os.path.join('..', '00_utils')))
append_path(os.path.abspath(os.path.join('..', '00_utils_training')))
append_path(os.path.abspath(os.path.join('..', '00_models')))

# preprocessing
DataPreprocessor = reload_module('preprocess', 'DataPreprocessor')

# generation of collocation points
import collocation_obj
Collocation = collocation_obj.Collocation

# ode solvers
import ode_solver_pyomo_opt
DirectODESolver = ode_solver_pyomo_opt.DirectODESolver

import nn_pyomo_base
NeuralODEPyomo = nn_pyomo_base.NeuralODEPyomo

class Trainer:
    def __init__(self, params_results, params_data, params_model, params_solver, params_ode = None, Ds_train = None, Ds_test = None):
        self.file_path = params_data['file_path']
        self.start_date = params_data['start_date']
        self.n_points, self.split = params_data['n_points'], params_data['split']
        self.n_days, self.m = params_data['n_days'], params_data['m']
        self.encoding = params_data['encoding']
        
        self.spacing = params_data.get('spacing', 'chebyshev')
        self.prev_hour = params_data.get('prev_hour', True)
        self.prev_week = params_data.get('prev_week', True)
        self.prev_year = params_data.get('prev_year', True)
        
        print(params_model)
        self.layer_sizes = params_model['layer_sizes']
        self.penalty = params_model['penalty']
        self.w_init_method = params_model['w_init_method']
                
        self.params_solver = params_solver
        self.params_ode = params_ode
        
        self.plot_directory = '../00_plots/pyomo'
        self.plot_collocation = params_results['plot_collocation']
        self.plot_odeint = params_results['plot_odeint']
        
        self.Ds_train = Ds_train
        self.Ds_test = Ds_test
    
    def clear_directory(self):
        """ Clear all files in the folder without deleting the folder itself. """
        folder_path = self.plot_directory
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) 
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    def save_trained_weights(self, description):
        weights = self.ode_model.extract_weights()
        formatted_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = f'../00_trained_wb/{description}_{formatted_time}.pkl'
        with open(path, 'wb') as file:
            pickle.dump(weights, file)
        print(f"Results saved to {path}")
    
    def train(self):
        print(f'Spacing type: {self.spacing}')
        data_loader = DataPreprocessor(self.file_path, start_date = self.start_date, 
                                       number_of_points = self.n_points, n_days = self.n_days, m = self.m, 
                                       feature_encoding = self.encoding, split = self.split, 
                                       prev_hour = self.prev_hour, prev_week = self.prev_week, prev_year = self.prev_year,
                                       spacing = self.spacing)
        
        data_subsample = data_loader.load_data()
        df_train, df_test = data_loader.preprocess_data(data_subsample)
        
        if self.Ds_train is None or self.Ds_test is None:
            self.Ds_train, self.Ds_test = data_loader.derivative_matrix() 
        
        ys = np.atleast_2d(df_train['y']).T
        ts = np.array(df_train['t'])
        Xs = np.atleast_2d(df_train.drop(columns=['y', 't']))
        
        print(f'w init method: {self.w_init_method}')
        self.ode_model = NeuralODEPyomo(y_observed = ys, 
                        t = ts, 
                        first_derivative_matrix = self.Ds_train, 
                        extra_input = Xs, 
                        y_init = ys,
                        layer_sizes = self.layer_sizes, act_func = "tanh", 
                        penalty_lambda_reg = self.penalty, 
                        time_invariant = True,
                        w_init_method = self.w_init_method, 
                        params = self.params_solver
                        )
        
        self.ode_model.build_model()
        result = self.ode_model.solve_model()
        u_model = self.ode_model.extract_solution().T
        self.termination = result['termination_condition']
        
        experiment_results = {}
        experiment_results['result'] = result
        experiment_results['times_elapsed'] = result['solver_time']
        
        # ---------------------------------------------------- ODEINT PREDICTION ------------------------------------------------
        y_pred = self.ode_model.neural_ode(ys[0], ts, (Xs, ts))
        experiment_results['mse_odeint'] = np.mean(np.square(np.squeeze(y_pred) - np.squeeze(ys)))
        
        # -------------------------------------------- COLLOCATION PREDICTION (TRAIN) ---------------------------------------------- 
        trained_weights_biases = self.ode_model.extract_weights()
            
        initial_state = ys[0][0]
        direct_solver = DirectODESolver(np.array(ts), self.layer_sizes, trained_weights_biases, initial_state, 
                                        D = self.Ds_train, 
                                        time_invariant=True, extra_input=np.array(Xs), params = self.params_ode)
        direct_solver.build_model()
        solver_info = direct_solver.solve_model()
        y_solution = direct_solver.extract_solution()     
        # experiment_results['mae_coll_ode'] = np.mean(np.abs(np.squeeze(y_solution) - np.squeeze(ys)))
        experiment_results['mse_coll_ode'] = np.mean(np.square(np.squeeze(y_solution) - np.squeeze(ys)))
        
        # ---------------------------------------- ODEINT & COLLOCATION PREDICTION (TEST) ----------------------------------------- 
        ys_test = np.atleast_2d(df_test['y']).T
        ts_test = np.array(df_test['t'])
        Xs_test = np.atleast_2d(df_test.drop(columns=['y', 't']))
        
        y_pred_test = self.ode_model.neural_ode(ys_test[0], ts_test, (Xs_test, ts_test))
        
        experiment_results['mse_odeint_test'] = np.mean(np.square(np.squeeze(y_pred_test) - np.squeeze(ys_test)))
        if self.plot_odeint:
            plt.figure(figsize=(10, 6))
            plt.plot(ts, ys, label='True Data', alpha = 1, color = 'green', ls = '--')
            plt.plot(ts_test, ys_test, alpha = 1, color = 'green', ls = '--')
            plt.plot(ts_test, y_pred_test, color='blue', label='Model Prediction (Test) -  Odeint', alpha = 1)
            plt.plot(ts, y_pred, color='#FF8C10', label='Model Prediction (Train) - Odeint', alpha = 1)
            plt.title(f"Collocation-based training & sequential predictions: True Data vs Model Prediction (Test)")
            plt.legend(loc ="lower right")
            plt.grid(True)
            plt.savefig(f'{self.plot_directory}/ode_solver_test_{self.start_date}.png', format='png')  
            plt.close() 
        
        y0_test = ys_test[0][0]
        direct_solver = DirectODESolver(ts_test, self.layer_sizes, trained_weights_biases, y0_test, 
                                        D = self.Ds_test,
                                        time_invariant=True, extra_input=np.array(Xs_test), params = self.params_ode)
        direct_solver.build_model()
        
        solver_info = direct_solver.solve_model()
        y_solution_test = direct_solver.extract_solution() 
        
        experiment_results['mse_coll_ode_test'] = np.mean(np.square(np.squeeze(y_solution_test) - np.squeeze(ys_test)))
        
        if self.plot_collocation:
            plt.figure(figsize=(10, 6))
            plt.plot(ts, ys, label='True Data', ls = '--', alpha = 1, color = 'green')
            plt.plot(ts_test, ys_test, ls = '--', color = 'green')
            plt.plot(ts, y_solution, color='blue', label='Model Prediction (Train) - collocation-based ODE', alpha = 1)
            plt.plot(ts_test, y_solution_test, color='#FF8C10', label='Model Prediction (Test) -  collocation-based ODE', alpha = 1)
            plt.title(f"Collocation-based training & collocation-based predictions: True Data vs Model Prediction")
            plt.legend(loc ="lower right")
            plt.grid(True)
            plt.savefig(f'{self.plot_directory}/collocation_solver_train_{self.start_date}.png', format='png')  
            plt.close() 
                
        return experiment_results
    
    