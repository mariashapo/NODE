import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# jax
import jax.numpy as jnp
import time

import sys
import os
import importlib
import pickle
import itertools

path_ = os.path.abspath(os.path.join('..', '00_utils'))
if path_ not in sys.path:
    sys.path.append(path_)

path_ = os.path.abspath(os.path.join('..', '00_models'))
if path_ not in sys.path:
    sys.path.append(path_)

path_ = os.path.abspath(os.path.join('..', '00_utils_training'))
if path_ not in sys.path:
    sys.path.append(path_)

import run_train_pyomo_rl

import logging
logging.basicConfig(level=logging.ERROR, filename='error_log.txt')

class ExperimentRunner:
    """
        sequence_len
        frequency
    
    """
    
    def __init__(self, start_date, optimization_aim, extra_inputs = {}):
        self.start_date = start_date
        self.opt_aim = optimization_aim
        self.extra_inputs = extra_inputs
        
        self.param_combinations = self.define_param_combinations()
        self.metrics = self.initialize_metrics()
        
        # generate default parameters for data, solver and ode
        # if not provided by the user
        self.ls = self.extra_inputs.get('ls', [6, 32, 1])
        self.penalty = self.extra_inputs.get('penalty', 1e-5)
        self.params_data = self.extra_inputs.get('params_data', self.default_data_params(start_date))
        self.params_solver = self.default_params_solver()
        self.params_ode = self.default_params_ode()
        
        self.sequence_len = self.extra_inputs.get('sequence_len', 5)
        self.frequency = self.extra_inputs.get('frequency', 2)
        self.date_sequences = ExperimentRunner.generate_dates(start_date, self.sequence_len, self.frequency)
        self.results_full = {}
        self.results_avg = {}
        
        self.params_results = {}
        self.params_results['plot_collocation'] = self.extra_inputs.get('plot_collocation', True)
        self.params_results['plot_odeint'] = self.extra_inputs.get('plot_odeint', False)
        
        importlib.reload(run_train_pyomo_rl)
        self.Trainer = run_train_pyomo_rl.Trainer
    
    def initialize_metrics(self):
        metrics = {
            'times_elapsed': [],
            'mse_odeint': [],
            'mse_coll_ode': [],
            'mse_odeint_test': [],
            'mse_coll_ode_test': []
        }
        self.metrics = metrics
        
    def collect_metrics(self, experiment_results):
        self.metrics['times_elapsed'].append(experiment_results['times_elapsed'])
        self.metrics['mse_odeint'].append(experiment_results['mse_odeint'])
        self.metrics['mse_coll_ode'].append(experiment_results['mse_coll_ode'])
        self.metrics['mse_odeint_test'].append(experiment_results['mse_odeint_test'])
        self.metrics['mse_coll_ode_test'].append(experiment_results['mse_coll_ode_test'])
        
        
    @staticmethod
    def compute_averages(metrics):
        # compute average of each metric and return as a new dictionary
        averages = {key: sum(values) / len(values) for key, values in metrics.items()}
        return averages
    
    def define_param_combinations(self):
        """
        Define the parameter combinations to be used in the optimization;
        based on the optimization aim.
        """
        if self.opt_aim == 'regularization':
            param_combinations = [0, 1e-7, 1e-5, 1e-3, 0.01, 0.1]
        elif self.opt_aim == 'tolerance':
            param_combinations = [1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
        elif self.opt_aim == 'tolerance_mix':
            tol = [1e-4, 1e-6, 1e-8]
            tol_inf_and_viol = [1e-2, 1e-4, 1e-6]
            tol_dual_inf = [10, 1, 1e-1]
            param_combinations = list(itertools.product(tol, tol_inf_and_viol, tol_inf_and_viol, tol_dual_inf))
        elif self.opt_aim == 'reg_tol':
            penalty_values = [0, 1e-7, 1e-5, 1e-3, 0.01, 0.1]
            tol_list = [1, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
            param_combinations = list(itertools.product(penalty_values, tol_list))
        elif self.opt_aim == 'input_features':
            param_combinations = self.extra_inputs
            if param_combinations == None:
                raise ValueError("extra_inputs must be provided for input_features optimization")
        elif self.opt_aim == 'collocation_method':
            param_combinations = ["chebyshev", "gauss_legendre", "gauss_radau", "gauss_lobatto"]
        elif self.opt_aim == 'convergence':
            iters = np.array([i for i in range(1,20)])
            param_combinations = iters*10
        elif self.opt_aim == 'default':
            param_combinations = [1]
        else:
            raise ValueError("optimization_aim not recognized")
        
        self.param_combinations = param_combinations
        return param_combinations

    @staticmethod
    def generate_dates(start_date, sequence_len = 5, frequency = 2):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        date_sequences = [start_date + timedelta(days=i*frequency) for i in range(sequence_len)]
        date_sequences_str = [date.strftime('%Y-%m-%d') for date in date_sequences]
        return date_sequences_str
    
    @staticmethod
    def default_data_params(start_date):
        print("Generating default parameters for data")
        params_data = {'file_path': '../00_data/df_train.csv', 'start_date': start_date, 
                'n_points': 400, 'split': 200, 'n_days': 1, 'm': 0, 
                'prev_hour': True, 'prev_week': True, 'prev_year': False,
                'spacing': 'gauss_radau',
                ''
                'encoding': {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'},}
        return params_data
    
    @staticmethod
    def default_params_solver():
        print("Generating default parameters for solver")
        params_solver = { "tol" : 1e-4, 
                         #"dual_inf_tol": 1e-4, #"compl_inf_tol": 1e-3, #"constr_viol_tol": 1e-5, 
                         "bound_relax_factor": 0.1, 
                         "acceptable_constr_viol_tol": 1e-15, "acceptable_dual_inf_tol": 1e-15, "acceptable_compl_inf_tol": 1e-15, 
                         "halt_on_ampl_error" : 'yes', "print_level": 1, "max_iter": 500, 'warm_start_init_point': 'yes'}
        return params_solver
    
    @staticmethod
    def default_params_ode():
        print("Generating default parameters for ode solver")
        params_ode = {"print_level": 1}
        return params_ode
    
    def update_prams(self, param_comb):
        if self.opt_aim == 'regularization':
            self.penalty = param_comb
        elif self.opt_aim == 'tolerance':
            self.params_solver['tol'] = param_comb
        elif self.opt_aim == 'tolerance_mix':
            self.params_solver['tol'] = param_comb[0]
            self.params_solver['tol_inf'] = param_comb[1]
            self.params_solver['tol_viol'] = param_comb[2]
            self.params_solver['tol_dual_inf'] = param_comb[3]
        elif self.opt_aim == 'reg_tol':
            self.params_solver['penalty'] = param_comb[0]
            self.params_solver['tol'] = param_comb[1]
        elif self.opt_aim == 'input_features':
            self.params_data['prev_hour'] = param_comb['prev_hour']
            self.params_data['prev_week'] = param_comb['prev_week']
            self.params_data['prev_year'] = param_comb['prev_year']
            self.params_data['m'] = param_comb['m']
            self.params_data['ls'] = param_comb['ls']
        elif self.opt_aim == 'collocation_method':
            self.params_data['spacing'] = param_comb
        elif self.opt_aim == 'convergence':
            self.params_solver['max_iter'] = param_comb
        elif self.opt_aim == 'default':
            pass
        else:
            raise ValueError("optimization_aim not recognized")
    
    def update_date(self, date, param_comb, file, iter):
        self.params_data['start_date'] = date
        print(f"Running iteration {iter} with parameters: {param_comb}")
        file.write(f"Running iteration {iter} with parameters: {param_comb}\n")
    
    def run(self):
        with open('results.txt', 'w'):
            pass
        file = open('results.txt', 'a')
        
        iter = 1
        for param_comb in self.param_combinations:
            
            self.update_prams(param_comb) # update parametes based on the optimization aim
            self.initialize_metrics()
            
            for date in self.date_sequences:
                self.update_date(date, param_comb, file, iter)
                self.params_model = {'layer_sizes': self.ls, 'penalty': self.penalty}
                
                try:
                    trainer = self.Trainer(self.params_results, self.params_data, self.params_model, self.params_solver, self.params_ode)
                    if iter == 1:
                        trainer.clear_directory()
                    experiment_results = trainer.train()
                except Exception as e:
                    print(f"Failed to complete training: {e}")
                    continue
                
                try:
                    self.results_full[(param_comb, date)] = experiment_results
                except Exception as e:
                    print(f"Failed to extract results: {e}")
                    continue                
                
                self.collect_metrics(experiment_results)
                file.write(f"param_comb: {param_comb}, date: {date}, results: {experiment_results}\n")
                file.flush() 
                
                iter += 1
                print (f"i: {iter}/{len(self.param_combinations)*len(self.date_sequences)}")
                
            try:
                self.results_avg[param_comb] = self.compute_averages(self.metrics)
            except Exception as e:
                print(f"Failed to compute averages: {e}")
                continue
            
        file.close()
            
    def save_results(self, description):
        formatted_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = f'results/{description}_{formatted_time}_full.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self.results_full, file)
        print(f"Results saved to {path}")
        
        path = f'results/{description}_{formatted_time}_avg.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self.results_avg, file)
        print(f"Results saved to {path}")