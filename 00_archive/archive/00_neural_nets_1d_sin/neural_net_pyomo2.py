import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, SolverFactory, value, RangeSet

import jax
import jax.numpy as jnp

from scipy.optimize import fsolve

import warnings

class NeuralODEPyomo2:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant = True, extra_input = None, 
                 penalty_lambda=0.1, max_iter=500, penalty_lambda_input = 0.001,
                 act_func="tanh", w_init_method="random", params = None, y_init = None):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda = penalty_lambda
        self.max_iter = max_iter
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.model = ConcreteModel()
        self.y_init = y_init
        self.w_init_method = w_init_method
        self.time_invariant = time_invariant
        self.extra_inputs = extra_input
        self.params = params
        self.observed_dim = None
        self.data_dim = None
        self.penalty_lambda_input = penalty_lambda_input,

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

    def build_model(self):
        # N = number of time steps
        # M = number of observed variables 
        # (eg. 2 for u and v)
        N, M = self.y_observed.shape 
        # rows
        self.data_dim = N
        # columns
        self.observed_dim = M
        
        model = self.model
        model.t_idx = RangeSet(0, N - 1)
        model.var_idx = RangeSet(0, M - 1)
        
        weight_bounds = (-100.0, 100.0)
        input_size = self.layer_sizes[0]
        layer1 = self.layer_sizes[1]
        
        model.W1 = pyo.Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        model.b1 = pyo.Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        
        if len(self.layer_sizes) == 3:
            output_size = self.layer_sizes[2]
            model.W2 = pyo.Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
            model.b2 = pyo.Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
            
        elif len(self.layer_sizes) == 4:
            layer2 = self.layer_sizes[2]
            output_size = self.layer_sizes[3]
            model.W2 = pyo.Var(range(layer2), range(layer1), initialize=0.1)
            model.b2 = pyo.Var(range(layer2), initialize=0.1)
            model.W3 = pyo.Var(range(output_size), range(layer2), initialize=0.1)
            model.b3 = pyo.Var(range(output_size), initialize=0.1)
        else:
            raise ValueError("layer_sizes should have exactly 3 elements: [input_size, hidden_size, output_size].")
        
        def func(y):
            nn_y = np.zeros_like(y)
            for i, y_ in enumerate(y):
                nn_y[i] = self.nn_output([y_], model)
            return self.first_derivative_matrix @ y - nn_y

        y = fsolve(func, self.y_observed)

        def _objective(m):
            data_fit = np.linalg.norm(y - self.y_observed, ord=2)
            reg = sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + \
                  sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + \
                  sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + \
                  sum(m.b2[j]**2 for j in range(self.layer_sizes[2]))

            if len(self.layer_sizes) == 4:
                reg += sum(m.W3[j, k]**2 for j in range(self.layer_sizes[3]) for k in range(self.layer_sizes[2])) + \
                       sum(m.b3[j]**2 for j in range(self.layer_sizes[3]))

            return data_fit + reg * self.penalty_lambda

        model.obj = Objective(rule=_objective, sense=pyo.minimize)
        self.model = model
    
    def nn_output(self, nn_input, m):

        if len(self.layer_sizes) == 3:
            hidden = np.dot(m.W1, nn_input) + m.b1
            epsilon = 1e-10
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
                
            outputs = np.dot(m.W2, hidden) + m.b2
        
        elif len(self.layer_sizes) == 4:
            hidden1 = np.dot(m.W1, nn_input) + m.b1
            
            if self.act_func == "tanh":
                hidden1 = [pyo.tanh(h) for h in hidden1]
            elif self.act_func == "sigmoid":
                hidden1 = [1 / (1 + pyo.exp(-h)) for h in hidden1]
            elif self.act_func == "softplus":
                hidden1 = [pyo.log(1 + pyo.exp(h)) for h in hidden1]
            
            hidden2 = np.dot(m.W2, hidden1) + m.b2
            
            if self.act_func == "tanh":
                hidden2 = [pyo.tanh(h) for h in hidden2]
            elif self.act_func == "sigmoid":
                hidden2 = [1 / (1 + pyo.exp(-h)) for h in hidden2]
            elif self.act_func == "softplus":
                hidden2 = [pyo.log(1 + pyo.exp(h)) for h in hidden2]
            
            outputs = np.dot(m.W3, hidden2) + m.b3
        else:
            raise ValueError("layer_sizes should have exactly 3 elements: [input_size, hidden_size, output_size].")
        
        return outputs

    def solve_model(self):
        solver = pyo.SolverFactory('ipopt')
        
        if self.max_iter:
            solver.options['max_iter'] = self.max_iter
        
        solver.options['halt_on_ampl_error'] = 'yes'
        
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        
        result = solver.solve(self.model, tee=True)
        
        # ------------------- Extract solver information -------------------
        solver_time = result.solver.time
        termination_condition = result.solver.termination_condition
        message = result.solver.message
        
        # ----------------- Extracted information in a dictionary ----------
        solver_info = {
            'solver_time': solver_time,
            'termination_condition': termination_condition,
            'message': message
        }
        
        print(solver_info)
        return solver_info

    def extract_solution(self):
        if self.observed_dim == 1:
            y = np.array([pyo.value(self.model.y[i]) for i in self.model.t_idx])
            return y
        elif self.observed_dim == 2:
            y1 = np.array([pyo.value(self.model.y1[i]) for i in self.model.t_idx])
            y2 = np.array([pyo.value(self.model.y2[i]) for i in self.model.t_idx])
            return [y1, y2]

    def extract_weights(self):
        weights = {}
        
        W1 = np.array([[pyo.value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([pyo.value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[pyo.value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([pyo.value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
        weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
        
        if len(self.layer_sizes) == 4:
            W3 = np.array([[pyo.value(self.model.W3[j, k]) for k in range(self.layer_sizes[2])] for j in range(self.layer_sizes[3])])
            b3 = np.array([pyo.value(self.model.b3[j]) for j in range(self.layer_sizes[3])])
            weights['W3'], weights['b3'] = W3, b3
        return weights

    def predict(self, input):
        weights = self.extract_weights()

        if len(self.layer_sizes) == 3:
            W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
            hidden = jnp.tanh(jnp.dot(W1, input) + b1)
            outputs = jnp.dot(W2, hidden) + b2
        elif len(self.layer_sizes) == 4:
            W1, b1, W2, b2, W3, b3 = weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3']
            hidden1 = jnp.tanh(jnp.dot(W1, input) + b1)
            hidden2 = jnp.tanh(jnp.dot(W2, hidden1) + b2)
            outputs = jnp.dot(W3, hidden2) + b3
            
        return outputs

    def mae(self, y_true, y_pred):
        mae_result = np.mean(np.abs(y_true - y_pred))
        return mae_result
    
    def mse(self, y_true, y_pred):
        mse_result = np.mean(np.squared(y_true - y_pred))
        return mse_result


if __name__ == "__main__":
    """ode_model = ODEOptimizationModel(y_observed, t, first_derivative_matrix)
    ode_model.build_model()
    ode_model.solve_model()
    u, v = ode_model.extract_solution()"""
