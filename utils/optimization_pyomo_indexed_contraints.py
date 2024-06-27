import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, SolverFactory, value, RangeSet
import jax
import jax.numpy as jnp

class ODEOptimizationModel:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, penalty_lambda=100, act_func="tanh", w_init_method="random", objective="mse", y_init=None, 
                 max_iter=None, max_cpu=None, tol=None, acceptable_tol=None, acceptable_iter=None, params=None, verbose=False, print_level=4):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix

        self.model = ConcreteModel()
        
        # model parameters
        self.penalty_lambda = penalty_lambda
        self.act_func = act_func
        self.layer_sizes = layer_sizes
        self.objective = objective
        
        # initialization parameters
        self.w_init_method = w_init_method
        self.y_init = y_init
        
        # solver parameters
        self.max_iter = max_iter
        self.verbose = verbose
        self.max_cpu = max_cpu
        self.tol = tol
        self.acceptable_tol = acceptable_tol
        self.acceptable_iter = acceptable_iter
        self.params = params
        
        # output parameters
        self.print_level = print_level

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

        if self.y_init is None:
            model.u = Var(model.t_idx, domain=pyo.Reals, initialize=0.1)
            model.v = Var(model.t_idx, domain=pyo.Reals, initialize=0.1)
        else:
            model.u = Var(model.t_idx, domain=pyo.Reals, initialize=dict(enumerate(self.y_init[0])))
            model.v = Var(model.t_idx, domain=pyo.Reals, initialize=dict(enumerate(self.y_init[1])))

        if len(self.layer_sizes) == 3:
            input_size, hidden_size, output_size = self.layer_sizes

            model.W1 = Var(range(hidden_size), range(input_size), initialize=0.1)
            model.b1 = Var(range(hidden_size), initialize=0.1)
            model.W2 = Var(range(output_size), range(hidden_size), initialize=0.1)
            model.b2 = Var(range(output_size), initialize=0.1)

        elif len(self.layer_sizes) == 4:
            input_size, hidden_size1, hidden_size2, output_size = self.layer_sizes

            model.W1 = Var(range(hidden_size1), range(input_size), initialize=0.1)
            model.b1 = Var(range(hidden_size1), initialize=0.1)
            model.W2 = Var(range(hidden_size2), range(hidden_size1), initialize=0.1)
            model.b2 = Var(range(hidden_size2), initialize=0.1)
            model.W3 = Var(range(output_size), range(hidden_size2), initialize=0.1)
            model.b3 = Var(range(output_size), initialize=0.1)

        penalty_terms = []

        def collocation_constraint_u(m, i):
            du_dt = sum(self.first_derivative_matrix[i, j] * m.u[j] for j in range(N))
            nn_u = self.nn_output_u(m.u[i], m.v[i], m)
            collocation_term = nn_u - du_dt
            penalty_terms.append(collocation_term**2)
            return collocation_term == 0

        def collocation_constraint_v(m, i):
            dv_dt = sum(self.first_derivative_matrix[i, j] * m.v[j] for j in range(N))
            nn_v = self.nn_output_v(m.u[i], m.v[i], m)
            collocation_term = nn_v - dv_dt
            penalty_terms.append(collocation_term**2)
            return collocation_term == 0

        model.ode_u = Constraint(range(1, N), rule=collocation_constraint_u)
        model.ode_v = Constraint(range(1, N), rule=collocation_constraint_v)

        def _objective(m):
            if self.objective == "mae":
                data_fit = sum(self.mae(self.y_observed[i], m.u[i], m.v[i]) for i in m.t_idx)
            elif self.objective == "mse":
                data_fit = sum((m.u[i] - self.y_observed[i, 0])**2 + (m.v[i] - self.y_observed[i, 1])**2 for i in m.t_idx)
            penalty = self.penalty_lambda * sum(penalty_terms)
            return penalty + data_fit

        model.obj = Objective(rule=_objective, sense=pyo.minimize)
        self.model = model

    def nn_output_u(self, u, v, m):
        return self.nn_output(u, v, m)[0]

    def nn_output_v(self, u, v, m):
        return self.nn_output(u, v, m)[1]

    def nn_output(self, u, v, m):
        inputs = [u, v]
        if len(self.layer_sizes) == 3:
            W1, b1, W2, b2 = m.W1, m.b1, m.W2, m.b2
            hidden = [sum(W1[j, k] * inputs[k] for k in range(self.layer_sizes[0])) + b1[j] for j in range(self.layer_sizes[1])]
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h)) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h)) for h in hidden]
            outputs = [sum(W2[j, k] * hidden[k] for k in range(self.layer_sizes[1])) + b2[j] for j in range(self.layer_sizes[2])]
        elif len(self.layer_sizes) == 4:
            W1, b1, W2, b2, W3, b3 = m.W1, m.b1, m.W2, m.b2, m.W3, m.b3
            hidden1 = [sum(W1[j, k] * inputs[k] for k in range(self.layer_sizes[0])) + b1[j] for j in range(self.layer_sizes[1])]
            if self.act_func == "tanh":
                hidden1 = [pyo.tanh(h) for h in hidden1]
            elif self.act_func == "sigmoid":
                hidden1 = [1 / (1 + pyo.exp(-h)) for h in hidden1]
            elif self.act_func == "softplus":
                hidden1 = [pyo.log(1 + pyo.exp(h)) for h in hidden1]
            hidden2 = [sum(W2[j, k] * hidden1[k] for k in range(self.layer_sizes[1])) + b2[j] for j in range(self.layer_sizes[2])]
            if self.act_func == "tanh":
                hidden2 = [pyo.tanh(h) for h in hidden2]
            elif self.act_func == "sigmoid":
                hidden2 = [1 / (1 + pyo.exp(-h)) for h in hidden2]
            elif self.act_func == "softplus":
                hidden2 = [pyo.log(1 + pyo.exp(h)) for h in hidden2]
            outputs = [sum(W3[j, k] * hidden2[k] for k in range(self.layer_sizes[2])) + b3[j] for j in range(self.layer_sizes[3])]
        return outputs

    def solve_model(self):
        solver = SolverFactory('ipopt', tee=self.verbose)
        solver.options['print_level'] = self.print_level
        
        if self.max_iter is not None:
            solver.options['max_iter'] = self.max_iter
        if self.max_cpu is not None:
            solver.options['max_cpu_time'] = self.max_cpu
        if self.tol is not None:
            solver.options['tol'] = self.tol
        if self.acceptable_tol is not None:
            solver.options['acceptable_tol'] = self.acceptable_tol
        if self.acceptable_iter is not None:
            solver.options['acceptable_iter'] = self.acceptable_iter
        
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        
        result = solver.solve(self.model, tee=self.verbose)
        
        # Extract solver information
        solver_time = result.solver.time
        termination_condition = result.solver.termination_condition
        message = result.solver.message
        
        # Extracted information in a dictionary
        solver_info = {
            'solver_time': solver_time,
            'termination_condition': termination_condition,
            'message': message
        }
        
        return solver_info
        
    def extract_solution(self):
        u = np.array([value(self.model.u[i]) for i in self.model.t_idx])
        v = np.array([value(self.model.v[i]) for i in self.model.t_idx])
        return u, v

    def extract_weights(self):
        weights = {}
        if len(self.layer_sizes) == 3:
            W1 = np.array([[value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
            b1 = np.array([value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
            W2 = np.array([[value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
            b2 = np.array([value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
            weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
        elif len(self.layer_sizes) == 4:
            W1 = np.array([[value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
            b1 = np.array([value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
            W2 = np.array([[value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
            b2 = np.array([value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
            W3 = np.array([[value(self.model.W3[j, k]) for k in range(self.layer_sizes[2])] for j in range(self.layer_sizes[3])])
            b3 = np.array([value(self.model.b3[j]) for j in range(self.layer_sizes[3])])
            weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3'] = W1, b1, W2, b2, W3, b3
        return weights

    def predict(self, u, v):
        weights = self.extract_weights()
        inputs = jnp.array([u, v])

        if len(self.layer_sizes) == 3:
            W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
            hidden = jnp.tanh(jnp.dot(W1, inputs) + b1)
            outputs = jnp.dot(W2, hidden) + b2
        elif len(self.layer_sizes) == 4:
            W1, b1, W2, b2, W3, b3 = weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3']
            hidden1 = jnp.tanh(jnp.dot(W1, inputs) + b1)
            hidden2 = jnp.tanh(jnp.dot(W2, hidden1) + b2)
            outputs = jnp.dot(W3, hidden2) + b3
        return outputs

    def mae(self, y_true, u, v):
        combined = np.vstack((u, v)).T
        mae_result = np.mean(np.abs(y_true - combined))
        return mae_result


if __name__ == "__main__":
    # Example usage (commented out):
    # ode_model = ODEOptimizationModel(y_observed, t, first_derivative_matrix, layer_sizes)
    # ode_model.build_model()
    # ode_model.solve_model()
    # u, v = ode_model.extract_solution()
    pass
