"""Module for running synthetic toy problem training for all three Neural ODE implementations."""
import numpy as np
import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import json
from pathlib import Path

from jax import random

from utils.data_generation import generate_ode_data
from utils.non_parametric_collocation import collocate_data
from utils.collocation_obj import Collocation
from models.nn_pyomo_base import NeuralODEPyomo as PyomoModel
from models.nn_jax_diffrax import NeuralODE as JaxDiffModel
from models.nn_pytorch import NeuralODE as PytorchModel
from models.ode_solver_pyomo_base import DirectODESolver # direct solver for post-training predictions

class TrainerToy:
    def __init__(self, params_data, model_type):
        self.N = params_data['N']
        self.N_test = params_data.get('N_test', max(1, self.N // 2))
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
        forcing_A = self.data_param.get('A', None)
        if self.model_type == 'pyomo':
            self.generate_nodes()
        else:
            self.nodes = jnp.linspace(self.start_time, self.end_time, self.N)
        
        # self.nodes will override t, start_time, end_time, spacing_type, n_points within generate_ode_data() function
        self.t, self.y, self.y_noisy, true_derivative = generate_ode_data(
            self.N, self.noise_level, self.ode_type, self.data_param, 
            initial_state = self.init_state, t = self.nodes, A = forcing_A)
        
        self.true_derivative = true_derivative
        
        # preserve the same span for test data and deduce test nodes
        self.start_time_test = max(self.nodes)
        self.end_time_test = max(self.nodes) + (max(self.nodes) - min(self.nodes))

        if self.model_type == 'pyomo':
            self.generate_nodes_test()
        else:
            # For non-Pyomo models, test window continues after the train window.
            self.nodes_test = jnp.linspace(self.start_time_test, self.end_time_test, self.N_test)

        self.init_state_test = self.y[-1]
        t_test, y_test, _, _ = generate_ode_data(
            self.N_test, self.noise_level, self.ode_type, self.data_param, 
            initial_state = self.init_state_test, t = self.nodes_test, A = forcing_A)
        
        self.t_test = t_test
        self.y_test = y_test
    
    def generate_nodes(self):
        collocation = Collocation(self.N, self.start_time, self.end_time, self.spacing_type)
        self.nodes = collocation.compute_nodes()
        self.collocation = collocation

    def generate_nodes_test(self):
        collocation_test = Collocation(self.N_test, self.start_time_test, self.end_time_test, self.spacing_type)
        self.nodes_test = collocation_test.compute_nodes()
        self.collocation_test = collocation_test

    def prepare_collocation(self):
        self.D = np.array(self.collocation.compute_derivative_matrix())
        self.D_test = np.array(self.collocation_test.compute_derivative_matrix())
        
    def estimate_derivative(self):
        _, est_sol = collocate_data(self.y_noisy, self.t, 'EpanechnikovKernel', bandwidth=0.5)
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
    def train(self, params_model, params_solver = None, seed = 42):
        if self.model_type == 'pyomo':
            self.train_pyomo(params_model, params_solver, seed)
        elif self.model_type == 'jax_diffrax':
            self.train_diffrax(params_model, params_solver, seed)
        elif self.model_type == 'pytorch':
            self.train_pytorch(params_model, params_solver, seed)
            
    def extract_results(self, detailed = False):
        if self.model_type == 'pyomo':
            return self.extract_results_pyomo(detailed)
        elif self.model_type == 'jax_diffrax':
            return self.extract_results_diffrax(detailed)
        elif self.model_type == 'pytorch':
            return self.extract_results_pytorch(detailed)
    
    #----------------------------------------------------------------PYOMO TRAINING--------------------------------------------------- 
    def prepare_train_params_pyomo(self, params_model):
        self.layer_widths = params_model['layer_widths']
        self.act_func = params_model['act_func']
        self.lambda_reg = params_model['penalty_lambda_reg']
        self.time_invar = params_model['time_invariant']
        self.w_init_method = params_model['w_init_method']
        self.params = params_model['params']
        self.pre_initialize = params_model.get('pre_initialize', True)
        self.reg_norm = params_model.get('reg_norm', True)
        self.skip_collocation = params_model.get('skip_collocation', np.inf)
        self.redirect_logs = params_model.get('redirect_logs', False)

    def train_pyomo(self, params_model, seed):
        
        self.prepare_train_params_pyomo(params_model)
        
        if not self.pre_initialize:
            self.est_sol = None
        
        self.seed = seed
        self.model = PyomoModel(
                        self.y_noisy, # pass noisy data
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
                        skip_collocation = self.skip_collocation,
                        seed = seed,
                        init_state=self.init_state
                        )
        
        self.model.build_model()
        result = self.model.solve_model(redirect_logs = self.redirect_logs)        
        self.time_elapsed = result['solver_time']
        self.termination = result['termination_condition']
        print(result)
        
        
        
    def extract_results_pyomo(self, detailed = False):
        direct_model_pred = self.model.extract_solution()
        # regenerate train data
        # ----------------------------------- these are the ODEINT predictions -----------------------------------
        odeint_pred = self.model.neural_ode(self.init_state, self.t)
        odeint_pred_test = self.model.neural_ode(self.init_state_test, self.t_test)
        
        mse_train = np.mean((self.y - odeint_pred)**2)
        mse_test = np.mean((self.y_test - odeint_pred_test)**2)
        
        if self.detailed or detailed:
            # -------------------------------------- COLLOCATION PREDICTION (TRAIN) --------------------------------------
            trained_weights_biases = self.model.extract_weights()
            direct_solver = DirectODESolver(
                self.t,
                self.layer_widths,
                trained_weights_biases,
                self.init_state,
                self.D,
                y_init_guess=odeint_pred,
                time_invariant=self.time_invar,
            )
            direct_solver.build_model(lower_bound=-10.0, upper_bound=10.0)
            direct_solver.solve_model()
            y_solution = direct_solver.extract_solution()     
            mse_train_coll = np.mean(np.square(np.squeeze(self.y) - np.squeeze(y_solution)))

            # -------------------------------------- COLLOCATION PREDICTION (TEST) --------------------------------------
            direct_solver = DirectODESolver(
                self.t_test,
                self.layer_widths,
                trained_weights_biases,
                self.init_state_test,
                self.D_test,
                y_init_guess=odeint_pred_test,
                time_invariant=self.time_invar,
            )
            direct_solver.build_model(lower_bound=-10.0, upper_bound=10.0)
            direct_solver.solve_model()
            y_solution_test = direct_solver.extract_solution()     
            mse_test_coll = np.mean(np.square(np.squeeze(self.y_test) - np.squeeze(y_solution_test)))

        # ------------------------------------------------ FIGURES ---------------------------------------------------
        # generate timestamp
        # stamp = datetime.now().strftime("%y_%m_%d_%H_%M")
        # seed_dir = f"results/{self.ode_type}_seed_{self.seed}"
        # os.makedirs(seed_dir, exist_ok=True)
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.t, self.y, label='True Data', alpha = 1, color = 'green')
        # plt.plot(self.t, self.y_noisy, label='Noisy Data', alpha = 1, color = 'green')
        # plt.plot(self.t, odeint_pred, color='#FF8C10', label='Model Prediction (Train) - Odeint', alpha = 1)
        # plt.plot(self.t, y_solution, color='blue', label='Model Prediction (Train) - Collocation', alpha = 1, ls = '--')
        # plt.title(f"Collocation-based training (DEV))")
        # plt.legend(loc ="lower right")
        # plt.grid(True)
        # plt.savefig(f"{seed_dir}/colloc_solver_train_{stamp}.png", format="png") 
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.plot(self.t_test, self.y_test, label='True Data', alpha = 1, color = 'green')
        # plt.plot(self.t_test, odeint_pred_test, color='#FF8C10', label='Model Prediction (Test) - Odeint', alpha = 1)
        # plt.plot(self.t_test, y_solution_test, color='blue', label='Model Prediction (Test) - Collocation', alpha = 1, ls = '--')
        # plt.title(f"Collocation-based training (DEV))")
        # plt.legend(loc ="lower right")
        # plt.grid(True)
        # plt.savefig(f"{seed_dir}/colloc_solver_train_{stamp}.png", format="png") 
        # plt.close()

        if self.detailed or detailed:
            results = {
                'time_elapsed': self.time_elapsed,
                'direct_model_pred': direct_model_pred,
                'odeint_pred': odeint_pred,
                'odeint_pred_test': odeint_pred_test,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'mse_train_coll': mse_train_coll,
                'mse_test_coll': mse_test_coll,
                'termination': self.termination,
                'seed': self.seed
            }
        else:
            results = {
                'time_elapsed': self.time_elapsed,
                'mse_train': mse_train,
                'mse_test': mse_test,
                'termination': self.termination,
                'seed': self.seed
            }
        
        return results

    def extract_pyomo_weights(self):
        return self.model.extract_weights()
    
    def extract_pyomo_solution(self):
        return self.model.extract_solution()
    
    #----------------------------------------------------------------DIFFRAX TRAINING---------------------------------------------------
    def prepare_train_params_diffrax(self, params_model):
        self.layer_widths = params_model['layer_widths']
        self.lambda_reg = params_model['penalty_lambda_reg']
        self.reg_norm = params_model.get('reg_norm', True)
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
        

    def train_diffrax(self, params_model, custom_params, seed):
        self.prepare_train_params_diffrax(params_model)
        
        rng = random.PRNGKey(seed)
        print(f"Using seed {seed} for JAX training.")
        self.model = JaxDiffModel(self.layer_widths, self.time_invar, act_func = self.act_func)
        # initialize the training state
        self.state = self.model.create_train_state(
            rng, self.lr, self.lambda_reg, self.rtol, self.atol, self.dt0, custom_params, reg_norm=self.reg_norm
        )
        
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
            max_iter = self.max_iter[0] if isinstance(self.max_iter, (list, tuple)) else self.max_iter
            self.state, losses_ = self.model.train(self.state, self.t, 
                                          self.y_noisy, self.init_state,
                                          num_epochs = max_iter,
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
        self.reg_norm = params_model.get('reg_norm', True)
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
        
    def train_pytorch(self, params_model, custom_params, seed = 42):
        self.prepare_train_params_pytorch(params_model)
        
        # Initialize the model
        # Normalize weight decay by param count if requested to mirror Pyomo
        wd = self.lambda_reg
        if self.reg_norm:
            total_params = sum(self.layer_widths[i] * self.layer_widths[i + 1] for i in range(len(self.layer_widths) - 1))
            total_params += sum(self.layer_widths[1:])  # biases
            if total_params > 0:
                wd = wd / total_params
        self.model = PytorchModel(self.layer_widths, self.lr, weight_decay=wd, custom_weights = custom_params, time_invariant = self.time_invar, seed = seed)
        
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
            max_iter = self.max_iter[0] if isinstance(self.max_iter, (list, tuple)) else self.max_iter
            losses_ = self.model.train_model(self.t, self.y_noisy, self.init_state,
                                num_epochs=max_iter,
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
    def _default_noise_level(config_path: str = "src/configs/config_pyomo_synth.json", fallback: float = 0.1) -> float:
        """Load default noise level from config; fallback if unavailable."""
        try:
            cfg = json.loads(Path(config_path).read_text())
            return float(cfg.get("data", {}).get("noise_level", fallback))
        except Exception:
            return fallback

    @staticmethod
    def load_trainer(type_, spacing_type="chebyshev", model_type = "pyomo", detailed = False, noise_level: float = None, A: float = None):
        noise = noise_level if noise_level is not None else TrainerToy._default_noise_level()
        data_params_ho = {
            'N': 200,
            'noise_level': noise,
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
            'noise_level': noise,
            'ode_type': "van_der_pol",
            'data_param': {"mu": 1, "omega": 1, **({"A": A} if A is not None else {})},
            'start_time': 0,
            'end_time': 15,
            'spacing_type': spacing_type,
            'initial_state': np.array([0.0, 1.0]),
            'detailed' : detailed
        }

        data_params_do = {
            'N': 200,
            'noise_level': noise,
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
