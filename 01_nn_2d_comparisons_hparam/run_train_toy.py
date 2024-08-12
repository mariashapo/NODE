import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import time
import matplotlib.pyplot as plt
import torch

from jax import random

path_ = os.path.abspath(os.path.join('..', '00_utils'))

if path_ not in sys.path:
    sys.path.append(path_)
    
path_ = os.path.abspath(os.path.join('..', '00_models'))

if path_ not in sys.path:
    sys.path.append(path_)

from data_generation import generate_ode_data
# pyomo
from nn_pyomo_base import NeuralODEPyomo as PyomoModel 
from non_parametric_collocation import collocate_data
from collocation_obj import Collocation
# jax-diffrax
from nn_jax_diffrax import NeuralODE as JaxDiffModel
# pytorch
from nn_pytorch import NeuralODE as PytorchModel

class PyomoTrainerToy:
    def __init__(self, params_data, model_type):
        self.N = params_data['N']
        self.noise_level = params_data['noise_level']
        self.ode_type = params_data['ode_type']
        self.data_param = params_data['data_param']
        self.start_time = params_data['start_time']
        self.end_time = params_data['end_time']
        self.spacing_type = params_data['spacing_type']
        self.init_state = params_data['initial_state']
        
        m = model_type
        if m in ['pyomo', 'jax_diffrax', 'pytorch']:
            self.model_type = m
        else:
            raise ValueError(f"Unsupported model type provided: {m}")
         
    def load_data(self):
        if self.model_type == 'pyomo':
            self.generate_nodes()
        else:
            self.nodes = jnp.linspace(self.start_time, self.end_time, self.N)
            
        self.t, self.y, self.y_noisy, true_derivative = generate_ode_data(
            self.N, self.noise_level, self.ode_type, self.data_param, 
            self.start_time, self.end_time, 
            initial_state = self.init_state, 
            t = self.nodes
            )
        
        # self.true_derivative = true_derivative
        
        test_end_time = self.end_time + (self.end_time - self.start_time)
        
        self.init_state_test = self.y[-1]
        t_test, y_test, _, _ = generate_ode_data(
            self.N*2, self.noise_level, self.ode_type, self.data_param, 
            self.end_time, test_end_time, 
            spacing_type = "uniform", 
            initial_state = self.init_state_test)
        
        self.t_test = t_test
        self.y_test = y_test
    
    def generate_nodes(self):
        collocation = Collocation(self.N, self.start_time, self.end_time, self.spacing_type)
        self.nodes = collocation.compute_nodes()
        self.collocation = collocation
       
    def prepare_collocation(self):
        self.D = np.array(self.collocation.compute_derivative_matrix(self.t))
        
    def estimate_derivative(self):
        est_der, est_sol = collocate_data(self.y_noisy, self.t, 'EpanechnikovKernel', bandwidth=0.5)
        self.est_sol = np.array(est_sol)
        
    def prepare_inputs(self):
        self.load_data()
        if self.model_type == 'jax':
            self.t = jnp.array(self.t)
            self.y = jnp.array(self.y)
            self.y_noisy = jnp.array(self.y_noisy)
            self.init_state = jnp.array(self.init_state)
            
            self.t_test = jnp.array(self.t_test)
            self.y_test = np.array(self.y_test)
            self.init_state_test = np.array(self.init_state_test)
        else:
            self.t = np.array(self.t)
            self.y = np.array(self.y)
            self.y_noisy = np.array(self.y_noisy)
            self.init_state = np.array(self.init_state)
            
            self.t_test = np.array(self.t_test)
            self.y_test = np.array(self.y_test)
            self.init_state_test = np.array(self.init_state_test)
            
        if self.model_type == 'pyomo':
            self.prepare_collocation()
            self.estimate_derivative()
    
    #----------------------------------------------------------------GENERAL PUBLIC FUNCTIONS ---------------------------------------------------        
    def train(self, params_model, params_solver = None):
        if self.model_type == 'pyomo':
            self.train_pyomo(params_model, params_solver)
        elif self.model_type == 'jax_diffrax':
            self.train_diffrax(params_model)
        elif self.model_type == 'pytorch':
            self.train_pytorch(params_model)
            
    def extract_results(self):
        if self.model_type == 'pyomo':
            return self.extract_results_pyomo()
        elif self.model_type == 'jax_diffrax':
            return self.extract_results_diffrax()
        elif self.model_type == 'pytorch':
            return self.extract_results_pytorch()
    
    #----------------------------------------------------------------PYOMO TRAINING--------------------------------------------------- 
    def prepare_train_params_pyomo(self, params_model):
        self.layer_widths = params_model['layer_widths']
        self.act_func = params_model['act_func']
        self.lambda_reg = params_model['penalty_lambda_reg']
        self.time_invar = params_model['time_invariant']
        self.w_init_method = params_model['w_init_method']
        self.params = params_model['params']

    def train_pyomo(self, params_model, params_solver = None):
        
        self.prepare_train_params_pyomo(params_model)
        
        self.model = PyomoModel(
                        self.y_noisy, # remember to pass noisy data
                        self.t, 
                        self.D,
                        self.layer_widths, 
                        act_func = self.act_func, 
                        y_init = self.est_sol, 
                        penalty_lambda_reg = self.lambda_reg, 
                        time_invariant = self.time_invar,
                        w_init_method = self.w_init_method, 
                        params = self.params
                        )
        
        self.model.build_model()
        result = self.model.solve_model()        
        self.time_elapsed = result['solver_time']
        self.termination = result['termination_condition']
        print(result)
        
    def extract_results_pyomo(self):
        direct_model_pred = self.model.extract_solution()
        # regenerate train data
        odeint_pred = self.model.neural_ode(self.init_state, self.t)
        odeint_pred_test = self.model.neural_ode(self.init_state_test, self.t_test)
        
        mse_train = np.mean((self.y - odeint_pred)**2)
        mse_test = np.mean((self.y_test - odeint_pred_test)**2)
        
        results = {
            'time_elapsed': self.time_elapsed,
            'direct_model_pred': direct_model_pred,
            'odeint_pred': odeint_pred,
            'odeint_pred_test': odeint_pred_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'termination': self.termination
        }
        
        return results

    #----------------------------------------------------------------DIFFRAX TRAINING---------------------------------------------------
    def prepare_train_params_diffrax(self, params_model):
        self.layer_widths = params_model['layer_widths']
        self.lambda_reg = params_model['penalty_lambda_reg']
        self.time_invar = params_model['time_invariant']
        self.max_iter = params_model['max_iter']
        
        self.lr = params_model.get('learning_rate', 1e-3)
        self.rtol = params_model.get('rtol', 1e-3)
        self.atol = params_model.get('atol', 1e-6)
        self.dt0 = params_model.get('dt0', 1e-3)
        self.pretrain_model = params_model.get('pretrain', False)
        self.verbose = params_model.get('verbose', True)
        # if pretrain is True, it should be passed as a list of 
        # the fractions of the data to be used for pre-training
        # [0.1, 0.5, 1]
        
    def train_diffrax(self, params_model):
        self.prepare_train_params_diffrax(params_model)
        
        rng = random.PRNGKey(42)
        self.model = JaxDiffModel(self.layer_widths, self.time_invar)
        # initialize the training state
        self.state = self.model.create_train_state(rng, self.lr, self.lambda_reg, self.rtol, self.atol, self.dt0)
        
        start_time = time.time()
        
        if self.init_state.ndim != 1:
            raise ValueError("Initial state for diffrax models must be a 1D array")
        
        if self.pretrain_model:
            for frac in self.pretrain_model:
                k = int(frac*len(self.t)) # calculate the number of data points to include
                self.state = self.model.train(self.state, self.t[:k], 
                                              self.y_noisy[:k], self.init_state,
                                              num_epochs = self.max_iter,
                                              verbose = self.verbose)
        else:
            self.state = self.model.train(self.state, self.t, 
                                          self.y_noisy, self.init_state,
                                          num_epochs = self.max_iter,
                                          verbose = self.verbose)
                
        self.time_elapsed = start_time - time.time()
        
    def extract_results_diffrax(self):
        odeint_pred = self.model.neural_ode(self.state.params, self.init_state, self.t, self.state)
        odeint_pred_test = self.model.neural_ode(
            self.state.params, self.init_state_test, self.t_test, self.state)
        
        mse_train = np.mean((self.y - odeint_pred)**2)
        mse_test = np.mean((self.y_test - odeint_pred_test)**2)
        
        results = {
            'time_elapsed': self.time_elapsed,
            'odeint_pred': odeint_pred,
            'odeint_pred_test': odeint_pred_test,
            'mse_train': mse_train,
            'mse_test': mse_test
        }
        
        return results
    
    #----------------------------------------------------------------PYTORCH TRAINING---------------------------------------------------
    def prepare_train_params_pytorch(self, params_model):
        self.layer_widths = params_model['layer_widths']
        self.lambda_reg = params_model['penalty_lambda_reg']
        self.time_invar = params_model['time_invariant']
        self.max_iter = params_model['max_iter']
        
        self.lr = params_model.get('learning_rate', 1e-3)
        self.rtol = params_model.get('rtol', 1e-3)
        self.atol = params_model.get('atol', 1e-4)
        self.dt0 = params_model.get('dt0', 1e-3)
        self.pretrain_model = params_model.get('pretrain', False)
        self.verbose = params_model.get('verbose', True)
        
    def train_pytorch(self, params_model):
        self.prepare_train_params_pytorch(params_model)
        
        self.model = PytorchModel(self.layer_widths, self.lr)
        # ode_model.train_model(t[:k], y_noisy[:k], y0, num_epochs = 1000)
        
        self.t = torch.tensor(self.t, dtype=torch.float32)
        self.y_noisy = torch.tensor(self.y_noisy, dtype=torch.float32)
        self.init_state = torch.tensor(self.init_state, dtype=torch.float32)
        
        if self.pretrain_model:
            for frac in self.pretrain_model:
                k = int(frac*len(self.t))
                self.model.train_model(self.t[:k], self.y_noisy[:k], self.init_state, 
                                 num_epochs = self.max_iter,
                                 rtol = self.rtol, atol = self.atol)
        else:
            self.model.train_model(self.t, self.y_noisy, self.init_state, 
                         num_epochs = self.max_iter, 
                         rtol = self.rtol, atol = self.atol)
            
    def extract_results_pytorch(self):
        odeint_pred = self.model.predict(self.t, self.init_state)
        self.t_test = torch.tensor(self.t_test, dtype=torch.float32)
        self.init_state_test = torch.tensor(self.init_state_test, dtype=torch.float32)
        odeint_pred_test = self.model.predict(self.t_test, self.init_state_test)
        
        mse_train = np.mean((self.y - odeint_pred.numpy())**2)
        mse_test = np.mean((self.y_test - odeint_pred_test.numpy())**2)
        
        results = {
            'odeint_pred': odeint_pred,
            'odeint_pred_test': odeint_pred_test,
            'mse_train': mse_train,
            'mse_test': mse_test
        }
        
        return results