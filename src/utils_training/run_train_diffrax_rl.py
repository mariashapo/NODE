import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
import os
import shutil
import importlib
import time

from utils.preprocess import DataPreprocessor_odeint as DataPreprocessor
from models.nn_jax_diffrax import NeuralODE as NeuralODE_JAX


class Trainer:
    def __init__(self, params_results, params_data, params_model, trained_wb = None):
        # data loading parameters
        self.file_path = params_data['file_path']
        self.start_date = params_data['start_date']
        self.n_points, self.split = params_data['n_points'], params_data['split']
        self.n_days, self.m = params_data['n_days'], params_data['m']
        self.encoding = params_data['encoding']
        
        # data feature parameters
        self.prev_hour = params_data.get('prev_hour', True)
        self.prev_week = params_data.get('prev_week', True)
        self.prev_year = params_data.get('prev_year', True)
        
        # model parameters
        self.layer_sizes = params_model['layer_sizes']
        self.penalty = params_model['penalty']
        self.learning_rate = params_model['learning_rate']
        self.num_epochs = params_model.get('num_epochs', 5000)
        self.pretrain = params_model.get('pretrain', False)
        # output parameters
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        self.plot_directory = os.path.join(repo_root, 'results', 'plots', 'diffrax')
        os.makedirs(self.plot_directory, exist_ok=True)
        self.plot = params_results['plot']
        self.log = params_results.get('log', False)
        self.split_time = params_results.get('split_time', False)
        
        self.trained_wb = trained_wb
        
        self.rng = random.PRNGKey(42)
        self.experiment_results = {}

        
    def clear_directory(self):
        """ Clear all files in the folder without deleting the folder itself. """
        folder_path = self.plot_directory
        os.makedirs(folder_path, exist_ok=True)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) 
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
    def train(self):
        # load data in
        data_loader = DataPreprocessor(self.file_path, start_date = self.start_date, 
                                       number_of_points = self.n_points, n_days = self.n_days, m = self.m, 
                                       feature_encoding = self.encoding, split = self.split, 
                                       prev_hour = self.prev_hour, prev_week = self.prev_week, prev_year = self.prev_year)
                                       # spacing = 'uniform', smooth = False)
        
        data_subsample = data_loader.load_data()
        df_train, df_test = data_loader.preprocess_data(data_subsample)
        
        # prepare inputs
        ys = jnp.atleast_2d(jnp.array(df_train['y'])).T
        ts = jnp.array(df_train['t'])
        Xs = jnp.array(df_train.drop(columns=['y', 't']))
        extra_args = (Xs, ts)
        y0 = jnp.array(ys[0])
        
        ys_test = jnp.atleast_2d(jnp.array(df_test['y'])).T
        ts_test = jnp.array(df_test['t'])
        Xs_test = jnp.array(df_test.drop(columns=['y', 't']))
        extra_args_test = (Xs_test, ts_test)
        y0_test = jnp.array(ys_test[0])
        
        
        if self.log:
            step = self.log
            self.log = {
                't': ts,
                'y': ys,
                'y_init': ys[0],
                'extra_args': extra_args,
                'epoch_recording_step' : step,
                't_test': ts_test,
                'y_test': ys_test,
                'y_init_test': ys_test[0],
                'extra_args_test': extra_args_test,
            }
        
        # prepare model
        node_model = NeuralODE_JAX(self.layer_sizes, time_invariant=True)
        state = node_model.create_train_state(self.rng, self.learning_rate, self.penalty, 
                                              rtol = 1e-3, atol = 1e-6, dt0 = 1e-3, custom_params = self.trained_wb)
        
        total_start = time.time()
        self.losses = [] 
        self.time_elapsed_parts = []
        
        if not self.pretrain:
            self.pretrain = [1]
            self.num_epochs = [self.num_epochs]
          
        for i, pretrain in enumerate(self.pretrain):
            stage_start = time.time()
            
            n = int(len(ts)* pretrain)
            state, losses = node_model.train(state, ts[:n] 
                                    , ys[:n], y0
                                    , num_epochs = self.num_epochs[i]
                                    , extra_args = extra_args[:n],
                                    log = self.log)
            
            
            self.losses.append(losses)
            if self.split_time:
                self.time_elapsed_parts.append(time.time() - stage_start)
        
        if not self.split_time:
            self.time_elapsed = time.time() - total_start
        else:
            self.time_elapsed = list(self.time_elapsed_parts)

        # ---------------------------------------------------- PREDICTION ------------------------------------------------
        y_train_pred = node_model.neural_ode(state.params, y0, ts, state, extra_args)
        
        self.experiment_results = {}
        # store scalar total, and keep per-stage breakdown if split_time requested
        if isinstance(self.time_elapsed, (list, tuple, np.ndarray)):
            runtime_total = float(np.sum(self.time_elapsed))
            self.experiment_results['times_elapsed_split'] = list(self.time_elapsed)
        else:
            runtime_total = float(self.time_elapsed)
        self.experiment_results['times_elapsed'] = runtime_total
        self.experiment_results['mse_diffrax'] = np.mean(np.square(np.squeeze(y_train_pred) - np.squeeze(ys)))
        
        y_test_pred = node_model.neural_ode(state.params, y0_test, ts_test, state, extra_args_test)      
        
        self.experiment_results['mse_diffrax_test'] = np.mean(np.square(np.squeeze(y_test_pred) - np.squeeze(ys_test)))  
        
        # ---------------------------------------------------- PLOTS ------------------------------------------------
        
        if self.plot:
            plt.figure(figsize=(10, 6))
            # true data
            plt.plot(ts, ys, alpha = 0.9, ls = '--', color = 'green', label='True Data')  
            plt.plot(ts_test, ys_test, alpha = 0.9, ls = '--', color = 'green')  
            
            # predictions
            plt.plot(ts, np.squeeze(y_train_pred), color='blue', label='Predicted Data (Train)', alpha = 1) 
            plt.plot(ts_test, np.squeeze(y_test_pred), color='#FF8C10', label='Predicted Data (Test)', alpha = 1) 
            
            plt.xlabel('Time')
            plt.ylabel('u(t)')
            plt.title(f"Diffrax Neural ODE: True Data vs Model Prediction")
            #plt.legend(loc ="lower right", bbox_to_anchor=(0.5, -0.3))
            plt.legend(loc ="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_directory, f'diffrax_solver_train_{self.start_date}.png'), format='png')  
            plt.close() 
            
        return self.experiment_results
    
    @staticmethod
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
