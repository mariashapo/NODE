import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Block, ConstraintList, Objective, SolverFactory, value, RangeSet

import jax
import jax.numpy as jnp

class ODEOptimizationModel:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, penalty_lambda=100, max_iter=500, act_func="tanh", w_init_method="random"):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda = penalty_lambda
        self.max_iter = max_iter
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.model = ConcreteModel()

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
        N = len(self.t)
        model = self.model
        model.t_idx = RangeSet(0, N - 1)

        model.u = pyo.Var(model.t_idx, domain=pyo.Reals, initialize=0.1)
        model.v = pyo.Var(model.t_idx, domain=pyo.Reals, initialize=0.1)

        # Create a Block to hold neural network parameters
        model.nn_block = Block()
        model.nn_block.layers = Block(range(len(self.layer_sizes) - 1))

        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]

            model.nn_block.layers[i].W = Var(range(output_size), range(input_size), initialize=lambda m, i, j: self.initialize_weights((output_size, input_size))[i, j])
            model.nn_block.layers[i].b = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i])

        penalty_terms = []
        model.ode = ConstraintList()
        for i in range(1, N):
            du_dt = sum(self.first_derivative_matrix[i, j] * model.u[j] for j in range(N))
            dv_dt = sum(self.first_derivative_matrix[i, j] * model.v[j] for j in range(N))

            nn_u, nn_v = self.nn_output(self.t[i], model.u[i], model.v[i], model)

            collocation_constraint_u = nn_u - du_dt
            collocation_constraint_v = nn_v - dv_dt

            model.ode.add(collocation_constraint_u == 0)
            model.ode.add(collocation_constraint_v == 0)

            penalty_terms.append((collocation_constraint_u)**2 + (collocation_constraint_v)**2)

        def _objective(m):
            data_fit = sum((m.u[i] - self.y_observed[i, 0])**2 + (m.v[i] - self.y_observed[i, 1])**2 for i in m.t_idx)
            penalty = self.penalty_lambda * sum(penalty_terms)
            return penalty + data_fit

        model.obj = Objective(rule=_objective, sense=pyo.minimize)
        self.model = model

    def nn_output(self, t, u, v, m):
        inputs = [t, u, v]
        for i in range(len(self.layer_sizes) - 1):
            layer = m.nn_block.layers[i]
            new_inputs = []
            for j in range(self.layer_sizes[i + 1]):
                neuron_input = sum(layer.W[j, k] * inputs[k] for k in range(self.layer_sizes[i])) + layer.b[j]
                if i < len(self.layer_sizes) - 2:
                    if self.act_func == "tanh":
                        neuron_input = pyo.tanh(neuron_input)
                    elif self.act_func == "sigmoid":
                        neuron_input = 1 / (1 + pyo.exp(-neuron_input))
                    elif self.act_func == "softplus":
                        neuron_input = pyo.log(1 + pyo.exp(neuron_input))
                new_inputs.append(neuron_input)
            inputs = new_inputs
        return inputs

    def solve_model(self, verbose=False):
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = self.max_iter
        solver.solve(self.model, tee=verbose)

    def extract_solution(self):
        u = np.array([pyo.value(self.model.u[i]) for i in self.model.t_idx])
        v = np.array([pyo.value(self.model.v[i]) for i in self.model.t_idx])
        return u, v

    def extract_weights(self):
        weights = {}
        for i in range(len(self.layer_sizes) - 1):
            layer = self.model.nn_block.layers[i]
            W = np.array([[pyo.value(layer.W[j, k]) for k in range(self.layer_sizes[i])] for j in range(self.layer_sizes[i + 1])])
            b = np.array([pyo.value(layer.b[j]) for j in range(self.layer_sizes[i + 1])])
            weights[f'layer_{i}'] = (W, b)
        return weights

    def predict(self, t, u, v):
        weights = self.extract_weights()
        inputs = jnp.array([t, u, v])

        for i in range(len(self.layer_sizes) - 1):
            W, b = weights[f'layer_{i}']
            inputs = jnp.dot(W, inputs) + b
            if i < len(self.layer_sizes) - 2:
                if self.act_func == "tanh":
                    inputs = jax.nn.tanh(inputs)
                elif self.act_func == "sigmoid":
                    inputs = jax.nn.sigmoid(inputs)
                elif self.act_func == "softplus":
                    inputs = jax.nn.softplus(inputs)
        return inputs

    def mae(self, y_true, u, v):
        combined = np.vstack((u, v)).T
        mae_result = np.mean(np.abs(y_true - combined))
        return mae_result
