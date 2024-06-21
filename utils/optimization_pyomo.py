import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, SolverFactory, value, RangeSet

class ODEOptimizationModel:
    """
    Constructs and solves an optimization model using neural networks to fit observed data points to an ODE system with collocation constraints.

    This function sets up a neural network within a Pyomo optimization framework to learn dynamics from observed data points, 
    minimizing the difference between observed states and predicted states by adjusting the neural network's weights and biases 
    subject to ODE constraints defined via a collocation matrix.

    Parameters:
    y_observed (np.ndarray): An array of observed data points with shape (N, 2), where N is the number of time points and 2 corresponds to the state variables u and v.
    t (np.ndarray): An array of time points corresponding to the observations in y_observed.
    first_derivative_matrix (np.ndarray): A matrix representing the first derivative constraints imposed by a collocation method on the ODE solution.
    penalty_lambda (float, optional): A regularization parameter used to scale the penalty terms in the objective function. Default is 100.
    max_iter (int, optional): The maximum number of iterations for the solver to run. Default is 500.
    act_func (str, optional): The activation function used in the neural network. Supported options are 'tanh', 'sigmoid', and 'softplus'. Default is 'tanh'.
    w_init_method (str, optional): Method to initialize weights; supported methods are 'random', 'xavier', and 'he'. Default is 'random'.

    Returns:
    ConcreteModel: A Pyomo ConcreteModel instance after solving. The model contains the solution, including the optimized values of state variables u and v, and the learned neural network parameters.

    Raises:
    ValueError: If an unsupported activation function or weight initialization method is specified.
    """
    def __init__(self, y_observed, t, first_derivative_matrix, penalty_lambda=100, max_iter=500, act_func="tanh", w_init_method="random"):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda = penalty_lambda
        self.max_iter = max_iter
        self.act_func = act_func
        self.w_init_method = w_init_method
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

    def nn_output(self, t, u, v, m):
        input_size = 3  # t, u, v
        hidden_size = 10
        output_size = 2  # du/dt and dv/dt

        hidden_layer_expr = [sum(m.W1[i, j] * (t if j == 0 else u if j == 1 else v) for j in range(input_size)) + m.b1[i] for i in range(hidden_size)]
        
        if self.act_func == "tanh":
            hidden_layer_output = [pyo.tanh(z) for z in hidden_layer_expr]
        elif self.act_func == "sigmoid":
            hidden_layer_output = [1 / (1 + pyo.exp(-z)) for z in hidden_layer_expr]
        elif self.act_func == "softplus":
            hidden_layer_output = [pyo.log(1 + pyo.exp(z)) for z in hidden_layer_expr]
        else:
            raise ValueError("Unsupported activation function. Use 'tanh', 'sigmoid', or 'softplus'.")

        output = [sum(m.W2[k, i] * hidden_layer_output[i] for i in range(hidden_size)) + m.b2[k] for k in range(output_size)]
        return output

    def build_model(self):
        N = len(self.t)
        model = self.model
        model.t_idx = RangeSet(0, len(self.y_observed)-1)
        model.dim_idx = RangeSet(0, 1)

        model.u = pyo.Var(model.t_idx, domain=pyo.Reals, initialize=0.1)
        model.v = pyo.Var(model.t_idx, domain=pyo.Reals, initialize=0.1)

        input_size = 3  # t, u, v
        hidden_size = 10  
        output_size = 2  # du/dt and dv/dt

        model.W1 = pyo.Var(range(hidden_size), range(input_size), initialize=lambda m, i, j: self.initialize_weights((hidden_size, input_size))[i, j])
        model.b1 = pyo.Var(range(hidden_size), initialize=lambda m, i: self.initialize_biases(hidden_size)[i])
        model.W2 = pyo.Var(range(output_size), range(hidden_size), initialize=lambda m, i, j: self.initialize_weights((output_size, hidden_size))[i, j])
        model.b2 = pyo.Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i])

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

    def solve_model(self):
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = self.max_iter
        solver.solve(self.model)

    def extract_solution(self):
        u = np.array([pyo.value(self.model.u[i]) for i in self.model.t_idx])
        v = np.array([pyo.value(self.model.v[i]) for i in self.model.t_idx])
        return u, v

if __name__ == "__main__":

    """ode_model = ODEOptimizationModel(y_observed, t, first_derivative_matrix)
    ode_model.build_model()
    ode_model.solve_model()
    u, v = ode_model.extract_solution()"""
    
