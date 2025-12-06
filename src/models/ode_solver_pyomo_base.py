import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet

class DirectODESolver:
    """
    Direct collocation-based solver treating the collocation as an optimization problem
    with 'soft' ODE constraints (residuals in the objective) and hard initial condition.
    """
    def __init__(self,
                 t,
                 layer_sizes,
                 trained_weights_biases,
                 initial_state,
                 D,
                 act_func="tanh",
                 time_invariant=True,
                 extra_input=None,
                 params=None,
                 y_init_guess=None):   # <--- NEW ARG
        self.t = t
        self.layer_sizes = layer_sizes
        self.initial_state = initial_state  
        self.act_func = act_func
        self.time_invariant = time_invariant
        self.extra_input = extra_input
        self.params = params

        # Ensure initial_state is a list of floats
        if isinstance(self.initial_state, np.ndarray):
            self.initial_state = self.initial_state.tolist()
        else:
            self.initial_state = [float(self.initial_state)]
        
        # Determine dimensions
        self.dimensions = len(self.initial_state)

        # Optional warm-start trajectory (shape (N, dim) or (N,))
        if y_init_guess is not None:
            arr = np.asarray(y_init_guess, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.shape[0] != len(t) or arr.shape[1] != self.dimensions:
                raise ValueError(
                    f"y_init_guess has shape {arr.shape}, "
                    f"expected ({len(t)}, {self.dimensions})"
                )
            self.y_init_guess = arr
        else:
            self.y_init_guess = None

        # Model weights
        self.W1 = trained_weights_biases['W1']
        self.W2 = trained_weights_biases['W2']
        self.b1 = trained_weights_biases['b1']
        self.b2 = trained_weights_biases['b2']

        if len(layer_sizes) == 4:
            self.W3 = trained_weights_biases['W3']
            self.b3 = trained_weights_biases['b3']

        # First derivative matrix
        self.D = D

    def build_model(self, lower_bound=-5.0, upper_bound=5.0):
        # Create a new model instance
        self.model = ConcreteModel()

        self.N = len(self.t)

        # Define sets for time points and dimensions
        self.model.t = RangeSet(0, self.N - 1)
        self.model.dimensions = RangeSet(0, self.dimensions - 1)

        # Initialization rule for y (warm-start if provided)
        if self.y_init_guess is not None:
            def y_init_rule(m, i, d):
                return float(self.y_init_guess[i, d])
        else:
            def y_init_rule(m, i, d):
                return 0.1

        # Define state variables over time and dimensions
        self.model.y = Var(
            self.model.t,
            self.model.dimensions,
            domain=pyo.Reals,
            initialize=y_init_rule,                  # <--- uses warm start
            bounds=(lower_bound, upper_bound)
        )

        # --- HARD INITIAL CONDITIONS (no slack) ---
        def init_condition_rule(m, d):
            return m.y[0, d] == self.initial_state[d]
        
        self.model.init_condition = Constraint(self.model.dimensions, rule=init_condition_rule)

        # Objective: sum of squared collocation residuals
        def _objective(m):
            total = 0.0
            for i in m.t:
                nn_input = [m.y[i, d] for d in m.dimensions]

                # add time and extra inputs if needed
                if not self.time_invariant:
                    nn_input.append(self.t[i])

                if self.extra_input is not None:
                    for input_array in self.extra_input.T:
                        nn_input.append(input_array[i])

                nn_output = self.nn_output(nn_input)

                for d in m.dimensions:
                    dy_dt = sum(self.D[i, j] * m.y[j, d] for j in m.t)
                    res = dy_dt - nn_output[d]
                    total += res**2
            return total
        
        self.model.obj = Objective(rule=_objective, sense=pyo.minimize)

    def nn_output(self, nn_input):
        epsilon = 1e-10
        if len(self.layer_sizes) == 3:
            hidden = np.dot(self.W1, nn_input) + self.b1
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
            else:
                raise ValueError("Unsupported activation function.")

            outputs = np.dot(self.W2, hidden) + self.b2  # length = dimensions
        elif len(self.layer_sizes) == 4:
            # For deeper networks
            hidden1 = np.dot(self.W1, nn_input) + self.b1
            if self.act_func == "tanh":
                hidden1 = [pyo.tanh(h) for h in hidden1]
            elif self.act_func == "sigmoid":
                hidden1 = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden1]
            elif self.act_func == "softplus":
                hidden1 = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden1]
            else:
                raise ValueError("Unsupported activation function.")

            hidden2 = np.dot(self.W2, hidden1) + self.b2
            if self.act_func == "tanh":
                hidden2 = [pyo.tanh(h) for h in hidden2]
            elif self.act_func == "sigmoid":
                hidden2 = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden2]
            elif self.act_func == "softplus":
                hidden2 = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden2]
            else:
                raise ValueError("Unsupported activation function.")

            outputs = np.dot(self.W3, hidden2) + self.b3
        else:
            raise ValueError("Only networks with 1 or 2 hidden layers are supported.")

        return outputs  # array/list of length 'dimensions'

    def solve_model(self, tee=False, logfile=None):
        # Solve the model using IPOPT
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(self.model, tee=tee, logfile=logfile)

        # Extract solver information
        solver_info = {
            'solver_time': result.solver.time,
            'termination_condition': result.solver.termination_condition,
            'message': result.solver.message
        }

        if tee:
            # Only print when explicitly requested
            print(solver_info)
        return solver_info

    def extract_solution(self):
        # Extract the solution for all time points and dimensions
        y_values = np.array([
            [value(self.model.y[i, d]) for d in self.model.dimensions]
            for i in self.model.t
        ])
        return y_values  # Shape: (N, dimensions)
