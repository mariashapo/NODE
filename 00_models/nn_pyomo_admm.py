import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, SolverFactory, value, RangeSet

import jax
import jax.numpy as jnp
import diffrax as dfx
import time

from ode_solver_pyomo_opt import DirectODESolver

np.random.seed(42)

class NeuralODEPyomoADMM:

    # --------------------------------------------- CLASS INITIALIZATION ------------------------------------------- #
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, rho = 1.0,
                 penalty_lambda_reg=0.01, penalty_lambda_smooth=0.0, act_func="tanh", w_init_method="random", params=None, y_init=None, test_data = None):

        # class parameters
        self.initialize_data_params(y_observed, t, extra_input, y_init, first_derivative_matrix)
        self.initialize_model_params(penalty_lambda_reg, act_func, w_init_method, 
                                layer_sizes, time_invariant, params, penalty_lambda_smooth, rho)
        self.initialize_admm_variables()
        
        self.iter = 0
        
        # model initialization
        self.model1, self.model2 = ConcreteModel(), ConcreteModel()
        self.create_submodels()
        
        if test_data is not None:
            self.test_data = test_data
            self.test_ys = test_data['y']
            self.test_ts = test_data['t']
            self.test_Xs = test_data['X']
            self.test_Ds = test_data['D']

    def initialize_data_params(self, y_observed, t, extra_input, y_init, first_derivative_matrix):        
        self.midpoint = len(t) // 2
        self.y_observed1, self.y_observed2 = y_observed[:self.midpoint], y_observed[self.midpoint:]
        self.t1, self.t2 = t[:self.midpoint], t[self.midpoint:]
        self.extra_input1, self.extra_input2 = self.split_if_not_none(extra_input)
        
        # initializion of the target variable, not initial condition
        self.y_init1, self.y_init2 = self.split_if_not_none(y_init) 
        self.D1, self.D2 = first_derivative_matrix
    
    def initialize_model_params(self, penalty_lambda_reg, act_func, w_init_method, 
                                layer_sizes, time_invariant, params, penalty_lambda_smooth, rho):
        self.w_init_method = w_init_method
        self.act_func = act_func
        self.layer_sizes = layer_sizes
        self.time_invariant = time_invariant
        self.params = params
        self.penalty_lambda_smooth = penalty_lambda_smooth
        self.penalty_lambda_reg = penalty_lambda_reg
        self.rho = rho

    def initialize_admm_variables(self):
        input_size, layer1, output_size = self.layer_sizes
        self.W1_consensus, self.b1_consensus = np.zeros((layer1, input_size)), np.zeros(layer1)
        self.W2_consensus, self.b2_consensus = np.zeros((output_size, layer1)), np.zeros(output_size)
        self.dual_W1, self.dual_b1 = np.zeros((layer1, input_size)), np.zeros(layer1)
        self.dual_W2, self.dual_b2 = np.zeros((output_size, layer1)), np.zeros(output_size)

    def split_if_not_none(self, arr):
        if arr is not None:
            return arr[:self.midpoint], arr[self.midpoint:]
        return None, None
    
    # --------------------------------------------- MODEL INITIALIZATION ---------------------------------------------- #
    def create_submodels(self):
        lower_bound, upper_bound = -5.0, 5.0
        
        self.data_dim, self.dimensions = self.y_observed1.shape
        self.model1.t, self.model2.t = RangeSet(0, self.data_dim - 1), RangeSet(0, self.data_dim - 1)
        
        self.initialize_target_variable(self.model1, self.y_init1, lower_bound, upper_bound)
        self.initialize_target_variable(self.model2, self.y_init2, lower_bound, upper_bound)
        
        self.initialize_nn_variables(self.model1)
        self.initialize_nn_variables(self.model2)
        
        self.model1.ode, self.model2.ode = ConstraintList(), ConstraintList()
        self.add_collocation_constraints()
        self.update_objective()
    
    def initialize_target_variable(self, model, y_init, lower_bound, upper_bound):
        if y_init is None:
            if self.dimensions == 1:
                model.y = pyo.Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
            elif self.dimensions == 2:
                model.y_d1 = pyo.Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
                model.y_d2 = pyo.Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        else:
            if self.dimensions == 1:
                model.y = pyo.Var(model.t, domain=pyo.Reals, initialize = y_init, bounds=(lower_bound, upper_bound))
            elif self.dimensions == 2:
                model.y_d1 = pyo.Var(model.t, domain=pyo.Reals, initialize = y_init[:, 0], bounds=(lower_bound, upper_bound))
                model.y_d2 = pyo.Var(model.t, domain=pyo.Reals, initialize = y_init[:, 1], bounds=(lower_bound, upper_bound))
    
    def initialize_nn_variables(self, model):
        input_size, layer1, output_size = self.layer_sizes
        weight_bounds = (-100.0, 100.0)
        model.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        model.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        model.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        model.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
        
        model.dual_W1 = np.zeros((layer1, input_size))
        model.dual_b1 = np.zeros(layer1)
        model.dual_W2 = np.zeros((output_size, layer1))
        model.dual_b2 = np.zeros(output_size)
        
    
    def initialize_weights(self, shape):
        if self.w_init_method == 'random':
            return np.random.randn(*shape) * 0.1
        elif self.w_init_method == 'xavier':
            return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))
        elif self.w_init_method == 'he':
            return np.random.randn(*shape) * np.sqrt(2 / shape[0])
        else:
            raise ValueError("Unsupported initialization method. Use 'random', 'xavier', or 'he'.")

    def initialize_biases(self, size):
        return np.random.randn(size) * 0.1
    
    def add_collocation_constraints(self):
        if self.dimensions == 1:
            # 1d case
            for i in range(1, len(self.data_dim)):
                dy_dt_m1 = sum(self.D1[i, j] * self.model1.y[j] for j in range(self.data_dim))
                dy_dt_m2 = sum(self.D2[i, j] * self.model2.y[j] for j in range(self.data_dim))
                
                nn_input_1, nn_input_2 = [self.model1.y[i]], [self.model2.y[i]]
                nn_input_1, nn_input_2 = self.add_time_and_extra_inputs(nn_input_1, nn_input_2, i)
                nn_y_m1, nn_y_m2 = self.nn_output(nn_input_1, self.model1), self.nn_output(nn_input_2, self.model2)
                self.model1.ode.add(nn_y_m1 == dy_dt_m1)
                self.model2.ode.add(nn_y_m2 == dy_dt_m2)
                        
        elif self.dimensions == 2:
            for i in range(1, self.data_dim):
                # print(i)
                dy_dt_d1_m1 = sum(self.D1[i, j] * self.model1.y_d1[j] for j in range(self.data_dim))
                dy_dt_d2_m1 = sum(self.D1[i, j] * self.model1.y_d2[j] for j in range(self.data_dim))

                dy_dt_d1_m2 = sum(self.D2[i, j] * self.model2.y_d1[j] for j in range(self.data_dim))
                dy_dt_d2_m2 = sum(self.D2[i, j] * self.model2.y_d2[j] for j in range(self.data_dim))                

                nn_input_1 = [self.model1.y_d1[i], self.model1.y_d2[i]]
                nn_input_2 = [self.model2.y_d1[i], self.model2.y_d2[i]]

                # Example assuming nn_output returns an array or tuple where each element corresponds to a dimension
                nn_d1_m1, nn_d2_m1 = self.nn_output(nn_input_1, self.model1)
                nn_d1_m2, nn_d2_m2 = self.nn_output(nn_input_2, self.model2)

                # Add two constrains for each timepoint to each of the models
                self.model1.ode.add((nn_d1_m1 == dy_dt_d1_m1))
                self.model1.ode.add((nn_d2_m1 == dy_dt_d2_m1))

                self.model2.ode.add((nn_d1_m2 == dy_dt_d1_m2))
                self.model2.ode.add((nn_d2_m2 == dy_dt_d2_m2))
                
        else:
            raise ValueError("Unsupported dimensions. Use 1 or 2.")
    
    def add_time_and_extra_inputs(self, nn_input_1, nn_input_2, i):
        if not self.time_invariant:
            nn_input_1.append(self.t1[i])
            nn_input_2.append(self.t2[i])
        if self.extra_input1 is not None:
            nn_input_1.extend(self.extra_input1[i])
        if self.extra_input2 is not None:
            nn_input_2.extend(self.extra_input2[i])
        return nn_input_1, nn_input_2

    # --------------------------------------------- MODEL UPDATES ---------------------------------------------- #
    
    def update_objective(self):
        if hasattr(self.model1, 'obj'):
            self.model1.del_component('obj')
        if hasattr(self.model2, 'obj'):
            self.model2.del_component('obj')
        self.model1.obj = Objective(rule=self.create_objective(self.y_observed1, self.t1), sense=pyo.minimize)
        self.model2.obj = Objective(rule=self.create_objective(self.y_observed2, self.t2), sense=pyo.minimize)

    def compute_regularization_term(self, m):
        layer1, input_size, output_size = self.layer_sizes[1], self.layer_sizes[0], self.layer_sizes[2]
        reg = (sum(m.W1[j, k]**2 for j in range(layer1) for k in range(input_size)) +
               sum(m.W2[j, k]**2 for j in range(output_size) for k in range(layer1)) +
               sum(m.b1[j]**2 for j in range(layer1)) +
               sum(m.b2[j]**2 for j in range(output_size)))
        return reg

    def compute_admm_penalty(self, m):
        layer1, input_size, output_size = self.layer_sizes[1], self.layer_sizes[0], self.layer_sizes[2]
        admm_penalty = (self.rho / 2) * (
            sum((m.W1[i, j] - self.W1_consensus[i, j] + m.dual_W1[i, j] / self.rho)**2 for i in range(layer1) for j in range(input_size)) +
            sum((m.b1[i] - self.b1_consensus[i] + m.dual_b1[i] / self.rho)**2 for i in range(layer1)) +
            sum((m.W2[i, j] - self.W2_consensus[i, j] + m.dual_W2[i, j] / self.rho)**2 for i in range(output_size) for j in range(layer1)) +
            sum((m.b2[i] - self.b2_consensus[i] + m.dual_b2[i] / self.rho)**2 for i in range(output_size))
        )
        return admm_penalty
    
    def create_objective(self, y_observed, t):
        """
        y_observed is passed as an argument when objective is created.
        _objective(m) takes the model as an argument - handled by Pyomo.
        """
        def _objective(m):
            if self.dimensions == 1:
                data_fit = sum((m.y[i] - y_observed[i])**2 for i in range(len(t)))
            elif self.dimensions == 2:
                data_fit = sum((m.y_d1[i] - y_observed[:,0][i])**2 + (m.y_d2[i] - y_observed[:,1][i])**2 for i in range(len(t)))
            
            reg = self.compute_regularization_term(m)
            admm_penalty = self.compute_admm_penalty(m) if self.iter >= 1 else 0
            return data_fit + self.penalty_lambda_reg * reg + admm_penalty   #+ self.penalty_lambda_smooth * reg_smooth
        
        return _objective
    
    # --------------------------------------------- MODEL OUTPUT ---------------------------------------------- # 

    def nn_output(self, nn_input, m):
        hidden = self.apply_activation_function(np.dot(m.W1, nn_input) + m.b1)
        outputs = np.dot(m.W2, hidden) + m.b2
        return outputs

    def apply_activation_function(self, hidden):
        epsilon = 1e-10
        if self.act_func == "tanh":
            return [pyo.tanh(h) for h in hidden]
        if self.act_func == "sigmoid":
            return [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
        if self.act_func == "softplus":
            return [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
        raise ValueError("Unsupported activation function. Use 'tanh', 'sigmoid', or 'softplus'.")

    # --------------------------------------------- PYOMO SOLVER ---------------------------------------------- # 

    def solve_model(self):
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value

        # Solve model1
        result1 = solver.solve(self.model1, tee=True)
        solver_info = self.extract_solver_info(result1, 'model1')

        # Solve model2
        result2 = solver.solve(self.model2, tee=True)
        solver_info.update(self.extract_solver_info(result2, 'model2'))

        # Update consensus and dual variables after both models are solved
        self.update_consensus_variables()
        self.update_dual_variables()
        return solver_info

    def extract_solver_info(self, result, model_name):
        return {model_name: {
            'solver_time': result.solver.time,
            'termination_condition': result.solver.termination_condition,
            'message': result.solver.message
        }}

    # --------------------------------------------- ADMM SOLVER ---------------------------------------------- #
    
    def admm_solve(self, iterations=50, tol_primal=1e-3, record=False):
        time_elapsed = 0
        for i in range(iterations):
            print('-' * 100)
            print(f"ADMM Iteration {i + 1}/{iterations}; {self.iter}")
            print('-' * 100)
            solver_info = self.solve_model()
            self.update_objective() # updates objectives for both models
            primal_residual = self.compute_primal_residual()
            print(f"Primal Residual: {primal_residual}")
            if record:
                time_elapsed += (solver_info['model1']['solver_time'] + solver_info['model2']['solver_time'])
                self.record_admm_info(time_elapsed)
            if primal_residual < tol_primal:
                print('*' * 100)
                print(f"Converged at iteration {i + 1}")
                print('*' * 100)
                return self.admm_info
            
            self.iter += 1
        print('*' * 100)
        print(f"Model did not converge. Primal Residual: {primal_residual}")
        if record:
            return self.admm_info
        print('*' * 100)
    
    # --------------------------------------------- RECORDING ---------------------------------------------- #
    def record_admm_info(self, time_elapsed):
        if not hasattr(self, 'admm_info'):
            self.admm_info = {'primal_residual': [], 'mse_diffrax': [], 'iter': [], 'time_elapsed': []}
            
        self.admm_info['primal_residual'].append(self.compute_primal_residual())
        
        # diffax predictions
        y_solution_1 = self.node_diffrax_pred(y0 = jnp.array(self.y_observed1[0]), t = jnp.array(self.t1), extra_input = None)
        y_solution_2 = self.node_diffrax_pred(y0 = jnp.array(self.y_observed2[0]), t = jnp.array(self.t2), extra_input = None)
        
        solution = np.squeeze(np.concatenate([y_solution_1, y_solution_2]))
        observed = np.squeeze(np.concatenate([self.y_observed1, self.y_observed2]))
        
        if solution.shape != observed.shape:
            raise ValueError("Solution and observed data do not have the same shape.")
        mse_diffrax = np.mean((solution - observed)**2)

        self.admm_info['mse_diffrax'].append(mse_diffrax)
        self.admm_info['iter'].append(self.iter)
        self.admm_info['time_elapsed'].append(time_elapsed)
              
    # --------------------------------------------- ADMM UPDATES ---------------------------------------------- # 
    def update_dual_variables(self):
        # Update dual variables for model1
        self.model1.dual_W1 += self.to_array(self.model1.W1) - self.W1_consensus
        self.model1.dual_b1 += self.to_vector(self.model1.b1) - self.b1_consensus
        self.model1.dual_W2 += self.to_array(self.model1.W2) - self.W2_consensus
        self.model1.dual_b2 += self.to_vector(self.model1.b2) - self.b2_consensus

        # Update dual variables for model2
        self.model2.dual_W1 += self.to_array(self.model2.W1) - self.W1_consensus
        self.model2.dual_b1 += self.to_vector(self.model2.b1) - self.b1_consensus
        self.model2.dual_W2 += self.to_array(self.model2.W2) - self.W2_consensus
        self.model2.dual_b2 += self.to_vector(self.model2.b2) - self.b2_consensus


    def update_consensus_variables(self):
        self.W1_consensus = (self.to_array(self.model1.W1) + self.to_array(self.model2.W1)) / 2
        self.b1_consensus = (self.to_vector(self.model1.b1) + self.to_vector(self.model2.b1)) / 2
        self.W2_consensus = (self.to_array(self.model1.W2) + self.to_array(self.model2.W2)) / 2
        self.b2_consensus = (self.to_vector(self.model1.b2) + self.to_vector(self.model2.b2)) / 2

    def to_array(self, pyomo_var):
        index_set = list(pyomo_var.index_set())
        shape = (max(i for i, j in index_set) + 1, max(j for i, j in index_set) + 1)
        return np.array([[pyomo_var[i, j].value for j in range(shape[1])] for i in range(shape[0])])

    def to_vector(self, pyomo_var):
        size = max(pyomo_var.index_set()) + 1
        return np.array([pyomo_var[i].value for i in range(size)])

    def compute_primal_residual(self):
        model1_vars = self.get_model_vars(self.model1)
        model2_vars = self.get_model_vars(self.model2)
        primal_residuals = [np.linalg.norm(model1_vars[var] - getattr(self, f"{var}_consensus")) + np.linalg.norm(model2_vars[var] - getattr(self, f"{var}_consensus"))
                            for var in ['W1', 'b1', 'W2', 'b2']]
        return sum(primal_residuals)
    
    def get_model_vars(self, model):
        return {
            'W1': self.to_array(model.W1),
            'b1': self.to_vector(model.b1),
            'W2': self.to_array(model.W2),
            'b2': self.to_vector(model.b2)
        }
    
    # --------------------------------------------- EXTRACT LEARNT SOLUTIONS ---------------------------------------------- # 
    def extract_solution(self):
        if self.dimensions == 1:
            y1 = np.array([value(self.model1.y[i]) for i in self.model1.t])
            y2 = np.array([value(self.model2.y[i]) for i in self.model2.t])
            return np.concatenate([y1, y2])
        elif self.dimensions == 2:
            y1_d1 = np.array([value(self.model1.y_d1[i]) for i in self.model1.t])
            y1_d2 = np.array([value(self.model1.y_d2[i]) for i in self.model1.t])
            y2_d1 = np.array([value(self.model2.y_d1[i]) for i in self.model2.t])
            y2_d2 = np.array([value(self.model2.y_d2[i]) for i in self.model2.t])
            y1 = np.vstack((y1_d1, y1_d2)).T  # Shape (len(t1), 2)
            y2 = np.vstack((y2_d1, y2_d2)).T  # Shape (len(t2), 2)
            return np.vstack((y1, y2))  # Concatenate along time axis


    def extract_derivative(self):
        if self.dimensions == 1:
            dy_dt_1 = [sum(self.D1[i, j] * pyo.value(self.model1.y[j]) for j in range(len(self.t1))) for i in range(len(self.t1))]
            dy_dt_2 = [sum(self.D2[i, j] * pyo.value(self.model2.y[j]) for j in range(len(self.t2))) for i in range(len(self.t2))]
            
        elif self.dimensions == 2:
            # Model 1 derivatives
            dy_dt_1_d1 = [sum(self.D1[i, j] * pyo.value(self.model1.y_d1[j]) for j in range(len(self.t1))) for i in range(len(self.t1))]
            dy_dt_1_d2 = [sum(self.D1[i, j] * pyo.value(self.model1.y_d2[j]) for j in range(len(self.t1))) for i in range(len(self.t1))]
            
            # Model 2 derivatives
            dy_dt_2_d1 = [sum(self.D2[i, j] * pyo.value(self.model2.y_d1[j]) for j in range(len(self.t2))) for i in range(len(self.t2))]
            dy_dt_2_d2 = [sum(self.D2[i, j] * pyo.value(self.model2.y_d2[j]) for j in range(len(self.t2))) for i in range(len(self.t2))]
            
            # Combine derivatives
            dy_dt_1 = np.array([dy_dt_1_d1, dy_dt_1_d2]).T  # Shape (len(t1), 2)
            dy_dt_2 = np.array([dy_dt_2_d1, dy_dt_2_d2]).T  # Shape (len(t2), 2)
            return dy_dt_1, dy_dt_2

    def extract_weights(self, m=None):
        if m is None:
            return {'W1': self.W1_consensus, 'b1': self.b1_consensus, 'W2': self.W2_consensus, 'b2': self.b2_consensus}
        W1 = np.array([[value(m.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([value(m.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[value(m.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([value(m.b2[j]) for j in range(self.layer_sizes[2])])
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    # --------------------------------------------- PREDICTIONS ---------------------------------------------- #
    def predict(self, input, weights):
        if weights == 'both':
            weights1, weights2 = self.extract_weights(self.model1), self.extract_weights(self.model2)
            W1, b1, W2, b2 = (weights1['W1'] + weights2['W1']) / 2, (weights1['b1'] + weights2['b1']) / 2, (weights1['W2'] + weights2['W2']) / 2, (weights1['b2'] + weights2['b2']) / 2
        elif weights == 'consensus':
            W1, b1, W2, b2 = self.W1_consensus, self.b1_consensus, self.W2_consensus, self.b2_consensus
        else:
            if weights == 'model1':
                weights1 = self.extract_weights(self.model1)
                W1, b1, W2, b2 = weights1['W1'], weights1['b1'], weights1['W2'], weights1['b2']
            elif weights == 'model2':
                weights2 = self.extract_weights(self.model2)
                W1, b1, W2, b2 = weights2['W1'], weights2['b1'], weights2['W2'], weights2['b2']
            else:
                raise ValueError("Not a valid weights argument.")
            
        hidden = jnp.tanh(jnp.dot(W1, input) + b1)
        outputs = jnp.dot(W2, hidden) + b2
        
        return outputs
    
    # --------------------------------------- SEQUENTIAL ODE SOLVER ------------------------------------------- #
    def node_diffrax_pred(self, y0, t, extra_input=None, weights='consensus'):
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)
            if args is not None:
                extra_inputs, t_all = args
                input = self.add_interpolated_inputs(input, extra_inputs, t, t_all)
            return self.predict(input, weights)
        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
        saveat = dfx.SaveAt(ts=t)
        solution = dfx.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[-1] - t[0], y0=y0, args=extra_input, stepsize_controller=stepsize_controller, saveat=saveat)
        return solution.ys
    
    def add_interpolated_inputs(self, input, extra_inputs, t, t_all):
        if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
            if extra_inputs.ndim == 2:
                interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                input = jnp.append(input, interpolated_inputs)
            elif extra_inputs.ndim == 1:
                interpolated_input = jnp.interp(t, t_all, extra_inputs)
                input = jnp.append(input, interpolated_input)
        else:
            input = jnp.append(input, extra_inputs)
        return input

    # --------------------------------------- COLLOCATION ODE SOLVER ------------------------------------------- #
    def node_collocation_pred(self, y0, t, D, extra_input = None, params = None):
        trained_params = self.extract_weights()
        
        direct_solver = DirectODESolver(t, self.layer_sizes, trained_params, y0, D, 
                                        act_func = self.act_func, time_invariant = self.time_invariant, 
                                        extra_input = extra_input, params = params)

        direct_solver.build_model()
        solver_info = direct_solver.solve_model()

        return direct_solver.extract_solution()

