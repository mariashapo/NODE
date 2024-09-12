import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import time
import matplotlib.pyplot as plt
import torch
import importlib

# to import activation functions
from flax import linen as nn 

from jax import random

path_ = os.path.abspath(os.path.join('..', '00_utils'))

if path_ not in sys.path:
    sys.path.append(path_)
    
path_ = os.path.abspath(os.path.join('..', '00_models'))

if path_ not in sys.path:
    sys.path.append(path_)

from data_generation import generate_ode_data
from non_parametric_collocation import collocate_data
from collocation_obj import Collocation

def reload_module(module_name, class_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return getattr(module, class_name)

PyomoModel = reload_module('nn_pyomo_base', 'NeuralODEPyomo')
JaxDiffModel = reload_module('nn_jax_diffrax', 'NeuralODE')
PytorchModel = reload_module('nn_pytorch', 'NeuralODE')


class TrainerToy:
    def __init__(self, params_data, model_type):
        self.N = params_data['N']
        self.noise_level = params_data['noise_level']
        self.ode_type = params_data['ode_type']
        self.data_param = params_data['data_param']
        self.start_time = params_data['start_time']
        self.end_time = params_data['end_time']
        self.spacing_type = params_data['spacing_type']
        self.init_state = params_data['initial_state']
        self.detailed = params_data.get('detailed', False)        
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
            min(self.nodes), max(self.nodes), 
            initial_state = self.init_state, 
            t = self.nodes
            )
        
        self.true_derivative = true_derivative
        
        test_end_time = max(self.nodes) + (max(self.nodes) - min(self.nodes))
        
        self.init_state_test = self.y[-1]
        t_test, y_test, _, _ = generate_ode_data(
            self.N*2, self.noise_level, self.ode_type, self.data_param, 
            max(self.nodes), test_end_time, 
            spacing_type = "uniform", 
            initial_state = self.init_state_test)
        
        self.t_test = t_test
        self.y_test = y_test
    
    def generate_nodes(self):
        collocation = Collocation(self.N, self.start_time, self.end_time, self.spacing_type)
        self.nodes = collocation.compute_nodes()
        self.collocation = collocation
       
    def prepare_collocation(self):
        self.D = np.array(self.collocation.compute_derivative_matrix())
        
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
            self.train_diffrax(params_model, params_solver)
        elif self.model_type == 'pytorch':
            self.train_pytorch(params_model, params_solver)
            
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
        self.pre_initialize = params_model.get('pre_initialize', True)
        self.reg_norm = params_model.get('reg_norm', False)
        self.skip_collocation = params_model.get('skip_collocation', np.inf)

    def train_pyomo(self, params_model, params_solver = None):
        
        self.prepare_train_params_pyomo(params_model)
        
        if not self.pre_initialize:
            self.est_sol = None
        
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
                        params = self.params,
                        reg_norm = self.reg_norm,
                        skip_collocation = self.skip_collocation
                        )
        
        self.model.build_model()
        result = self.model.solve_model()        
        self.time_elapsed = result['solver_time']
        self.termination = result['termination_condition']
        print(result)
        
    def extract_results_pyomo(self, detailed = False):
        direct_model_pred = self.model.extract_solution()
        # regenerate train data
        odeint_pred = self.model.neural_ode(self.init_state, self.t)
        odeint_pred_test = self.model.neural_ode(self.init_state_test, self.t_test)
        
        mse_train = np.mean((self.y - odeint_pred)**2)
        mse_test = np.mean((self.y_test - odeint_pred_test)**2)
        
        if self.detailed or detailed:
            results = {
                'time_elapsed': self.time_elapsed,
                'direct_model_pred': direct_model_pred,
                'odeint_pred': odeint_pred,
                'odeint_pred_test': odeint_pred_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'termination': self.termination
            }
        else:
            results = {
                'time_elapsed': self.time_elapsed,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'termination': self.termination
            }
        
        return results

    def extract_pyomo_weights(self):
        return self.model.extract_weights()
    
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
        self.log = params_model.get('log', False)
        self.split_time = params_model.get('split_time', False)
        self.act_func = params_model.get('act_func', 'tanh')
        
        if self.log:
            self.log = {
                't': self.t,
                'y': self.y,
                'y_init': self.init_state,
                'extra_args': None,
                'epoch_recording_step' : self.log,
                't_test': self.t_test,
                'y_test': self.y_test,
                'y_init_test': self.init_state_test,
                'extra_args_test': None
            }
        
        if self.act_func == 'tanh':
            self.act_func = jax.nn.tanh
        elif self.act_func == 'relu':
            self.act_func = jax.nn.relu
        elif self.act_func == 'sigmoid':
            self.act_func = jax.nn.sigmoid
        else:
            raise ValueError(f"Unsupported activation function provided: {self.act_func}")
        

    def train_diffrax(self, params_model, custom_params):
        self.prepare_train_params_diffrax(params_model)
        
        rng = random.PRNGKey(42)
        self.model = JaxDiffModel(self.layer_widths, self.time_invar, act_func = self.act_func)
        # initialize the training state
        self.state = self.model.create_train_state(rng, self.lr, self.lambda_reg, self.rtol, self.atol, self.dt0, custom_params)
        
        start_time = time.time()
        
        if self.init_state.ndim != 1:
            raise ValueError("Initial state for diffrax models must be a 1D array")
        self.losses = []
        
        if self.pretrain_model:
            if self.log or self.split_time:
                start_time = time.time()
                self.time_elapsed = []
            for i, frac in enumerate(self.pretrain_model):
                k = int(frac*len(self.t)) # calculate the number of data points to include
                self.state, losses_ = self.model.train(self.state, self.t[:k], 
                                              self.y_noisy[:k], self.init_state,
                                              num_epochs = self.max_iter[i],
                                              verbose = self.verbose,
                                              log = self.log)
                self.losses.append(losses_)
                
                if self.log or self.split_time:
                    self.time_elapsed.append(time.time() - start_time)
                    start_time = time.time()  # Reset start time for next segment
        else:
            self.state, losses_ = self.model.train(self.state, self.t, 
                                          self.y_noisy, self.init_state,
                                          num_epochs = self.max_iter,
                                          verbose = self.verbose,
                                          log = self.log)
            self.losses.append(losses_)
        
        if not (self.log or self.split_time) or not self.pretrain_model:        
            self.time_elapsed = time.time() - start_time
            
        
    def extract_results_diffrax(self, detailed = False):
        odeint_pred = self.model.neural_ode(self.state.params, self.init_state, self.t, self.state)
        odeint_pred_test = self.model.neural_ode(
            self.state.params, self.init_state_test, self.t_test, self.state)
        
        mse_train = np.mean((self.y - odeint_pred)**2)
        mse_test = np.mean((self.y_test - odeint_pred_test)**2)
        
        if self.detailed or detailed:
            results = {
                'time_elapsed': self.time_elapsed,
                'odeint_pred': odeint_pred,
                'odeint_pred_test': odeint_pred_test,
                'mse_train': mse_train,
                'mse_test': mse_test
            }
        else:
            results = {
                'time_elapsed': self.time_elapsed,
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
        self.log = params_model.get('log', False)
        self.split_time = params_model.get('split_time', False)
        
        if self.log:
            self.log = {
                't': torch.tensor(self.t, dtype=torch.float32), 
                'y': torch.tensor(self.y, dtype=torch.float32),
                'y_init': torch.tensor(self.init_state, dtype=torch.float32),
                'extra_args': None,
                'epoch_recording_step' : self.log,
                't_test': torch.tensor(self.t_test, dtype=torch.float32),
                'y_test':  torch.tensor(self.y_test, dtype=torch.float32),
                'y_init_test': torch.tensor(self.init_state_test, dtype=torch.float32),
                'extra_args_test': None
            }
        
    def train_pytorch(self, params_model, custom_params):
        self.prepare_train_params_pytorch(params_model)
        
        # Initialize the model
        self.model = PytorchModel(self.layer_widths, self.lr, custom_weights = custom_params)
        
        # Convert data to appropriate tensor format
        self.t = torch.tensor(self.t, dtype=torch.float32)
        self.y_noisy = torch.tensor(self.y_noisy, dtype=torch.float32)
        self.init_state = torch.tensor(self.init_state, dtype=torch.float32)
        
        start_time = time.time()
        self.losses = []
        
        if self.pretrain_model:
            if self.log or self.split_time:
                start_time = time.time()
                self.time_elapsed = []
                
            for i, frac in enumerate(self.pretrain_model):
                k = int(frac * len(self.t))
                losses_ = self.model.train_model(self.t[:k], self.y_noisy[:k], self.init_state,
                                    num_epochs=self.max_iter[i],
                                    rtol=self.rtol, atol=self.atol, log = self.log)
                self.losses.append(losses_)
                
                if self.log or self.split_time:
                    self.time_elapsed.append(time.time() - start_time)
                    start_time = time.time()  # Reset start time for next segment
        else:
            losses_ = self.model.train_model(self.t, self.y_noisy, self.init_state,
                                num_epochs=self.max_iter,
                                rtol=self.rtol, atol=self.atol, log = self.log)
            
            self.losses.append(losses_)
            
        if not (self.log or self.split_time) or not self.pretrain_model:     
            self.time_elapsed = time.time() - start_time
            
            
    def extract_results_pytorch(self, detailed = False):
        odeint_pred = self.model.predict(self.t, self.init_state)
        self.t_test = torch.tensor(self.t_test, dtype=torch.float32)
        self.init_state_test = torch.tensor(self.init_state_test, dtype=torch.float32)
        odeint_pred_test = self.model.predict(self.t_test, self.init_state_test)
        
        mse_train = np.mean((self.y - odeint_pred.numpy())**2)
        mse_test = np.mean((self.y_test - odeint_pred_test.numpy())**2)
        
        if self.detailed or detailed:
            results = {
                'time_elapsed': self.time_elapsed,
                'odeint_pred': odeint_pred,
                'odeint_pred_test': odeint_pred_test,
                'mse_train': mse_train,
                'mse_test': mse_test
            }
        else:
            results = {
                'time_elapsed': self.time_elapsed,
                'mse_train': mse_train,
                'mse_test': mse_test
            }
        
        return results
    
    # default parameters for toy datasets
    @staticmethod
    def load_trainer(type_, spacing_type="chebyshev", model_type = "pyomo", detailed = False):
        data_params_ho = {
            'N': 200,
            'noise_level': 0.2,
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
            'noise_level': 0.1,
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
            'noise_level': 0.1,
            'ode_type': "damped_oscillation",
            'data_param': {"damping_factor": 0.1, "omega_squared": 1},
            'start_time': 0,
            'end_time': 10,
            'spacing_type': spacing_type,
            'initial_state': np.array([0.0, 1.0]),
            'detailed' : detailed
        }

        if type_ == "ho":
            p_ = data_params_ho
        elif type_ == "vdp":
            p_ = data_params_vdp
        elif type_ == "do":
            p_ = data_params_do
        else:
            raise ValueError(f"Invalid type {type_}")

        if (model_type != 'pyomo' and model_type != 'jax_diffrax' and model_type != 'pytorch'):
            raise ValueError(f"model_type should be pyomo or jax_diffrax")
            
        trainer = TrainerToy(p_, model_type = model_type)
        trainer.prepare_inputs()
        return trainer