import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, SolverFactory, value, RangeSet

import jax.numpy as jnp
from jax.experimental.ode import odeint

import warnings

class NeuralODEPyomo:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, 
                 penalty_lambda_reg=0.1, penalty_lambda_input=0.001, constraint_penalty=0.1, 
                 act_func="tanh", w_init_method="random", params=None, y_init=None,
                 constraint = "l1"):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda_reg = penalty_lambda_reg
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.model = ConcreteModel()
        self.y_init = y_init
        self.time_invariant = time_invariant
        self.extra_inputs = extra_input
        self.params = params
        self.observed_dim = None
        self.data_dim = None
        self.penalty_lambda_input = penalty_lambda_input
        self.constraint_penalty = constraint_penalty
        self.constraint = constraint

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
        N, M = self.y_observed.shape 
        self.data_dim = N
        self.observed_dim = M
        
        model = self.model
        model.t_idx = RangeSet(0, N - 1)
        model.var_idx = RangeSet(0, M - 1)

        lower_bound = -5.0
        upper_bound = 5.0

        def y_init_rule(m, i, j):
            return self.y_init[i, j] if self.y_init is not None else 0.1
        
        model.y = Var(model.t_idx, model.var_idx, domain=pyo.Reals, initialize=y_init_rule, bounds=(lower_bound, upper_bound))

        weight_bounds = (-100.0, 100.0)
        input_size = self.layer_sizes[0]
        layer1 = self.layer_sizes[1]
        output_size = self.layer_sizes[2]
        
        model.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        model.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        model.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        model.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

        @model.Constraint(model.t_idx)
        def ode_constraint(m, i):
            #if i == 0:
                #return pyo.Constraint.Skip
            dy_dt = sum(self.first_derivative_matrix[i, j] * m.y[j, 0] for j in m.t_idx)
            nn_input = [m.y[i, 0]]  
            if not self.time_invariant:
                nn_input.append(self.t[i])
            if self.extra_inputs is not None:
                for input in self.extra_inputs.T:
                    nn_input.append(input[i])
                    
            nn_y = self.nn_output(nn_input, m)
            if self.constraint == "l2":
                return ((nn_y - dy_dt)**2 == 0)
            elif self.constraint == "l1":
                return ((nn_y == dy_dt))
            #return (nn_y - dy_dt) <= self.constraint_penalty
            #return (nn_y - dy_dt)**2 == 0
            
        @model.Objective(sense=pyo.minimize)
        def objective_rule(m):
            data_fit = sum((m.y[i, 0] - self.y_observed[i, 0])**2 for i in m.t_idx)
            penalty = sum((m.y[i, 0] - self.y_init[i, 0])**2 for i in range(N)) if self.y_init is not None else 0
            reg = sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + \
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + \
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + \
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2]))
            return data_fit + reg * self.penalty_lambda_reg + self.penalty_lambda_input * penalty**2

        self.model = model 

    def nn_output(self, nn_input, m):
        hidden = np.dot(m.W1, nn_input) + m.b1
        epsilon = 1e-10
        if self.act_func == "tanh":
            hidden = [pyo.tanh(h) for h in hidden]
        elif self.act_func == "sigmoid":
            hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
        elif self.act_func == "softplus":
            hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
        outputs = np.dot(m.W2, hidden) + m.b2
        return outputs

    def solve_model(self):
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(self.model, tee=True)
        solver_info = {
            'solver_time': result.solver.time,
            'termination_condition': result.solver.termination_condition,
            'message': result.solver.message
        }
        print(solver_info)
        return solver_info

    def extract_solution(self):
        y = np.array([value(self.model.y[i, 0]) for i in self.model.t_idx])
        return y

    def extract_weights(self):
        weights = {}
        W1 = np.array([[value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
        weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
        return weights
    
    def predict(self, input):
        """
        Outputs the predicted rate of change of the observed variables.
        """
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
    
    def neural_ode(self, y0, t, extra_args = None):
        
        def func(y, t, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)
    
            if args is not None:
                extra_inputs, t_all = args
            
            if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                
                if extra_inputs.ndim == 2:
                    # we have multiple datapoints
                    index = jnp.argmin(jnp.abs(t_all - t))
                    for extra_input in extra_inputs[index]:
                            input = jnp.append(input, extra_input)
                            
                elif extra_inputs.ndim == 1:
                    # we have a single datapoint so no need to slice the index
                    for extra_input in extra_inputs:
                            input = jnp.append(input, extra_input)
                    
            else: # if a single value, simply append it
                input = jnp.append(input, extra_inputs)
        
            # call the predict function to simulate the ODE
            result = self.predict(input)
            return result
    
        return odeint(func, y0, t, extra_args)