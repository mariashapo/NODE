import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import pickle

# jax
import jax.numpy as jnp
from jax import random
import time
import itertools

# stats
from statsmodels.api import tsa # time series analysis
import statsmodels.api as sm

# collocation
import sys
import os
import importlib

p_ = os.path.abspath(os.path.join('..', '00_utils'))
if p_ not in sys.path:
    sys.path.append(p_)
    
p_ = os.path.abspath(os.path.join('..', '00_models'))
if p_ not in sys.path:
    sys.path.append(p_)
    
p_ = os.path.abspath(os.path.join('..', '00_utils_training'))
if p_ not in sys.path:
    sys.path.append(p_)

import run_train_diffrax_rl


class ExperimentRunner:
    def __init__(self, start_date, optimization_aim, extra_inputs = {}):
        self.start_date = start_date
        self.opt_aim = optimization_aim
        self.extra_inputs = extra_inputs
        
        self.params_data = self.extra_inputs.get('params_data', ExperimentRunner.default_data_params(start_date))
        self.params_model = self.extra_inputs.get('params_model', ExperimentRunner.default_model_params())
        
        if 'params_sequence' in self.extra_inputs.keys():
            self.sequence_len = self.extra_inputs['params_sequence']['sequence_len']
            self.frequency = self.extra_inputs['params_sequence']['frequency']
        else:
            self.sequence_len = 5
            self.frequency = 2
            
        self.date_sequences = ExperimentRunner.generate_dates(start_date, self.sequence_len, self.frequency)
        self.results_full = {}
        self.results_avg = {}
        
        self.params_results = {}
        if 'params_results' in self.extra_inputs.keys():
            self.params_results['plot'] = self.extra_inputs['params_results'].get('plot', True)
            self.params_results['log'] = self.extra_inputs['params_results'].get('log', False)
            self.params_results['split_time'] = self.extra_inputs['params_results'].get('split_time', False)
        else:
            self.params_results['plot'] = True
            self.params_results['log'] = False
            self.params_results['split_time'] = False
        
        self.param_combinations = self.define_param_combinations()
        self.metrics = self.initialize_metrics()
        
        # reload the training module
        importlib.reload(run_train_diffrax_rl)
        self.Trainer = run_train_diffrax_rl.Trainer
    
    def initialize_metrics(self):
        metrics = {
            'times_elapsed': [],
            'mse_diffrax': [],
            'mse_diffrax_test': []
        }
        self.metrics = metrics
        
    def collect_metrics(self, experiment_results):
        self.metrics['times_elapsed'].append(experiment_results['times_elapsed'])
        self.metrics['mse_diffrax'].append(experiment_results['mse_diffrax'])
        self.metrics['mse_diffrax_test'].append(experiment_results['mse_diffrax_test'])  
    
    def define_param_combinations(self):
        """
        Define the parameter combinations to be used in the optimization;
        based on the optimization aim.
        """
        if self.opt_aim == 'regularization':
            param_combinations = [0, 1e-7, 1e-5, 1e-3, 0.01, 0.1]
        elif self.opt_aim == 'input_features':
            param_combinations = self.extra_inputs
            if param_combinations == None:
                raise ValueError("extra_inputs must be provided for input_features optimization")
        elif self.opt_aim == 'default':
            param_combinations = [1]
        elif self.opt_aim == 'network_size':
            in_layer = 6
            layer_sizes = [[in_layer, 16, 1], [in_layer, 32, 1], [in_layer, 64, 1], [in_layer, 128, 1],
                           [in_layer, 16, 16, 1], [in_layer, 32, 32, 1]]
            regularization = [0, 1e-7, 1e-5]
            num_epochs = [5000, 7500, 10000]
            param_combinations = list(itertools.product(layer_sizes, regularization, num_epochs))
        elif self.opt_aim == 'convergence':
            param_combinations = [1]
            if not self.params_results['log']:
                raise ValueError("log must be set to True for convergence optimization")
        else:
            raise ValueError("optimization_aim not recognized")
        
        self.param_combinations = param_combinations
        return param_combinations
    
    def update_params(self, param_comb):
        if self.opt_aim == 'regularization':
            self.penalty = param_comb
        elif self.opt_aim == 'input_features':
            self.params_data['prev_hour'] = param_comb['prev_hour']
            self.params_data['prev_week'] = param_comb['prev_week']
            self.params_data['prev_year'] = param_comb['prev_year']
            self.params_data['m'] = param_comb['m']
            self.params_data['ls'] = param_comb['ls']
        elif self.opt_aim == 'convergence':
            pass
        elif self.opt_aim == 'network_size':
            self.params_model['layer_sizes'] = param_comb[0]
            self.params_model['penalty'] = param_comb[1]
            
            if len(self.params_model['num_epochs']) > 1:
                self.params_model['num_epochs'][-1] = param_comb[2]
            else:
                self.params_model['num_epochs'] = param_comb[2]
                
        elif self.opt_aim == 'default':
            pass
        else:
            raise ValueError("optimization_aim not recognized")
  
    
    @staticmethod
    def default_data_params(start_date):
        print("Generating default parameters for data")
        params_data = {'file_path': '../00_data/df_train.csv', 'start_date': start_date, 
                'n_points': 400, 'split': 200, 'n_days': 1, 'm': 0, 
                'prev_hour': True, 'prev_week': True, 'prev_year': False,
                'spacing': 'uniform',
                'encoding': {'settlement_date': 't', 'temperature': 'var1', 'hour': 'var2', 'nd': 'y'}}
        return params_data
    
    @staticmethod
    def compute_averages(metrics):
        # compute average of each metric and return as a new dictionary
        averages = {key: sum(values) / len(values) for key, values in metrics.items()}
        return averages

    @staticmethod
    def default_model_params():
        print("Generating default parameters for model")
        params_model = {'layer_sizes': [8, 64, 64, 1], 'penalty': 1e-5, 'learning_rate': 1e-3, 'num_epochs': [2000, 5000], 'pretrain': [0.2, 1]}
        return params_model
    
    @staticmethod
    def generate_dates(start_date, sequence_len = 5, frequency = 2):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        date_sequences = [start_date + timedelta(days=i*frequency) for i in range(sequence_len)]
        date_sequences_str = [date.strftime('%Y-%m-%d') for date in date_sequences]
        return date_sequences_str

    @staticmethod
    def convert_lists_in_tuple(param_tuple):
        """
        Converts all list elements in a tuple to string representations,
        keeping all other elements unchanged.
        """
        
        return tuple(str(item) if isinstance(item, list) else item for item in param_tuple)
        
    def update_date(self, date, param_comb, file, iter):
        self.params_data['start_date'] = date
        print(f"Running iteration {iter} with parameters: {param_comb}")
        file.write(f"Running iteration {iter} with parameters: {param_comb}\n")

    def run(self):
        with open('results_diffrax.txt', 'w'):
            pass
        file = open('results_diffrax.txt', 'a')
        iter = 1
        
        if self.params_results['log']:
            self.losses = []
            
        for param_comb in self.param_combinations:
            self.initialize_metrics()
            self.update_params(param_comb)
            
            for date in self.date_sequences:
                self.update_date(date, param_comb, file, iter)
                
                try:
                    trainer = self.Trainer(self.params_results, self.params_data, self.params_model)
                    if iter == 1:
                        trainer.clear_directory()
                    experiment_results = trainer.train()
                except Exception as e:
                    print(f"Failed to complete training: {e}")
                    file.write(f"Error in iteration {iter}: {e}\n")
                    continue
                
                param_comb = ExperimentRunner.convert_lists_in_tuple(param_comb)
                
                try:
                    self.results_full[(param_comb, date)] = experiment_results
                except Exception as e:
                    print(f"Failed to save results: {e}")
                    file.write(f"Error in iteration {iter}: {e}\n")
                    continue
                
                if self.params_results['log']:
                    self.losses.append(trainer.losses)

                self.collect_metrics(trainer.experiment_results)
                file.write(f"param_comb: {param_comb}, date: {date}, results: {experiment_results}\n")
                file.flush() 
                print (f"Iteration i: {iter}/{len(self.param_combinations)*len(self.date_sequences)} completed")
                iter += 1
        
            try:
                if self.opt_aim != 'convergence':
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
        
        path = f'results/{description}_{formatted_time}_avg.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self.results_avg, file)
        print(f"Results saved to {path}")