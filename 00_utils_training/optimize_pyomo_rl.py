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
    extra_inputs: dict. Expected keys:
        layer_sizes; default: [6, 32, 1]
        penalty; default: 1e-5
        params_data; default: generated by default_data_params(start_date)
    
    """
    
    def __init__(self, start_date, optimization_aim, extra_inputs = {}):
        self.start_date = start_date
        self.opt_aim = optimization_aim
        self.extra_inputs = extra_inputs
        
        # generate default parameters for data, solver and ode
        # if not provided by the user
        
        self.initialize_inputs()
        self.date_sequences = ExperimentRunner.generate_dates(start_date, self.sequence_len, self.frequency)
        self.results_full = {}
        self.results_avg = {}
        
        self.params_results = {}
        self.params_results['plot_collocation'] = self.extra_inputs.get('plot_collocation', True)
        self.params_results['plot_odeint'] = self.extra_inputs.get('plot_odeint', False)
        
        self.params_model = {'layer_sizes': self.ls, 'penalty': self.penalty}
        
        if optimization_aim == 'convergence':
            self.convergence_step = self.extra_inputs.get('convergence_steps', 5)
        
        self.param_combinations = self.define_param_combinations()
        self.metrics = self.initialize_metrics()
        
        importlib.reload(run_train_pyomo_rl)
        self.Trainer = run_train_pyomo_rl.Trainer
        
        if 'param_combinations' in self.extra_inputs.keys():
            self.param_combinations = self.extra_inputs['param_combinations']
    
    def initialize_inputs(self):
        if 'params_model' in self.extra_inputs.keys():
            self.ls = self.extra_inputs['params_model'].get('layer_sizes', [6, 32, 1])
            self.penalty = self.extra_inputs['params_model'].get('penalty', 1e-5)
        else:
            print('Using default parameters for model')
            self.ls = [6, 32, 1]
            self.penalty = 1e-5
        
        if 'params_sequence' in self.extra_inputs.keys():
            self.sequence_len = self.extra_inputs['params_sequence']['sequence_len']
            self.frequency = self.extra_inputs['params_sequence']['frequency']
        else:
            print('Using default parameters for sequence')
            self.sequence_len = 5
            self.frequency = 2
        
        # not using 'get' to be explicit & generate print statements if default values are used  
        if 'params_data' in self.extra_inputs.keys():
            self.params_data = self.extra_inputs['params_data']
        else:
            self.params_data = self.default_data_params(self.start_date)
        
        if 'params_solver' in self.extra_inputs.keys():
            self.params_solver = self.extra_inputs['params_solver']
        else:
            self.params_solver = self.default_params_solver()
        
        self.params_ode = self.default_params_ode()
    
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
            tol_inf_and_viol = [1e-2, 1e-4]
            #tol_dual_inf = [10, 1, 1e-1]
            bound_relax_factor = [0.1, 1e-4, 1e-8]
            param_combinations = list(itertools.product(tol, tol_inf_and_viol, tol_inf_and_viol, bound_relax_factor))
            
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
            iters = np.array([i for i in range(1, 100)])
            iters *= self.convergence_step
            param_combinations = np.concatenate((np.array([1]), iters))
            self.optimal = {d:False for d in self.date_sequences}
            
        elif self.opt_aim == 'default':
            param_combinations = [1]
            
        elif self.opt_aim == 'network_size':
            sizes = [[6, 16, 1], [6, 32, 1], [6, 64, 1], [6, 128, 1]]
            reg = [1e-7, 1e-6, 1e-5]
            tol = [1e-8, 1e-6, 1e-4]
            param_combinations = list(itertools.product(sizes, reg, tol))
        else:
            raise ValueError("optimization_aim not recognized")
        
        self.param_combinations = param_combinations
        return param_combinations
    
    def update_prams(self, param_comb):
        if self.opt_aim == 'regularization':
            self.penalty = param_comb
            
        elif self.opt_aim == 'tolerance':
            self.params_solver['tol'] = param_comb
            
        elif self.opt_aim == 'tolerance_mix':
            self.params_solver['tol'] = param_comb[0]
            self.params_solver['constr_viol_tol'] = param_comb[1]
            self.params_solver['dual_inf_tol'] = param_comb[2]
            self.params_solver['bound_relax_factor'] = param_comb[3]
            
        elif self.opt_aim == 'reg_tol':
            self.penalty = param_comb[0]
            self.params_solver['tol'] = param_comb[1]
            self.params_solver['constr_viol_tol'] = param_comb[1]
            
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
            
        elif self.opt_aim == 'network_size':
            self.ls = param_comb[0]
            self.penalty = param_comb[1]
            self.params_solver['tol'] = param_comb[2]
            
        elif self.opt_aim == 'default':
            pass
        else:
            raise ValueError("optimization_aim not recognized")
        
        self.params_model = {'layer_sizes': self.ls, 'penalty': self.penalty}
        
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
        
    def update_date(self, date, param_comb, file, iter):
        self.params_data['start_date'] = date
        print(f"Running iteration {iter} with parameters: {param_comb}")
        file.write(f"Running iteration {iter} with parameters: {param_comb}\n")
        
    @staticmethod
    def convert_lists_in_tuple(param_tuple):
        """
        Converts all list elements in a tuple to string representations,
        keeping all other elements unchanged.
        """
        
        return tuple(str(item) if isinstance(item, list) else item for item in param_tuple)
    
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
                
                if self.opt_aim == 'convergence' and self.optimal[date]:
                    break
                
                try:
                    trainer = self.Trainer(self.params_results, self.params_data, self.params_model, self.params_solver, self.params_ode)
                    if iter == 1:
                        trainer.clear_directory()
                    experiment_results = trainer.train()
                    print(f"message: {trainer.termination}")
                    if self.opt_aim == 'convergence' and 'optimal' in trainer.termination:
                        print(f"Optimal solution for {date} found in iteration {param_comb}")
                        self.optimal[date] = True
                except Exception as e:
                    print(f"Failed to complete training: {e}")
                    continue
                
                try:
                    param_comb = ExperimentRunner.convert_lists_in_tuple(param_comb)
                    self.results_full[(param_comb, date)] = experiment_results
                except Exception as e:
                    print(f"Failed to extract results: {e}")
                    continue                
                
                self.collect_metrics(experiment_results)
                file.write(f"param_comb: {param_comb}, date: {date}, results: {experiment_results}\n")
                file.flush() 
                
                print (f"Iteration i: {iter}/{len(self.param_combinations)*len(self.date_sequences)} completed")
                iter += 1
            
            if self.opt_aim != 'convergence':    
                # no need to compute averages when recording training losses
                try:
                    if self.opt_aim == 'network_size':
                        key_name = (str(param_comb[0]), param_comb[1], date)
                        self.results_avg[key_name] = ExperimentRunner.compute_averages(self.metrics)
                    else:
                        self.results_avg[param_comb] = ExperimentRunner.compute_averages(self.metrics)
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
        
        if self.opt_aim != 'convergence':
            path = f'results/{description}_{formatted_time}_avg.pkl'
            with open(path, 'wb') as file:
                pickle.dump(self.results_avg, file)
            print(f"Results saved to {path}")