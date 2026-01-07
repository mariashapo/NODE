import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, SolverFactory, value, RangeSet

class DirectODESolver:
    """
    Direct collocation-based solver treating the collocation as an optimization problem with 'soft' constraints.
    """
    def __init__(self, t, layer_sizes, trained_weights_biases, initial_state, D, 
                 act_func="tanh", time_invariant=True, extra_input=None,
                 params=None):
        
        self.t = t
        self.layer_sizes = layer_sizes
        self.initial_state = initial_state  

        self.initial_state = np.atleast_1d(np.array(self.initial_state, dtype=float))
        self.dimensions = self.initial_state.shape[0]
        
        self.act_func = act_func
        self.time_invariant = time_invariant
        self.extra_input = extra_input
        self.params = params

        # model weights
        self.W1 = trained_weights_biases['W1']
        self.W2 = trained_weights_biases['W2']
        self.b1 = trained_weights_biases['b1']
        self.b2 = trained_weights_biases['b2']
        
        # derivative matrix
        self.D = D
        self.model = ConcreteModel()
        
    def build_model(self):
        self.N = len(self.t)
        
        lower_bound = -5.0
        upper_bound = 5.0

        # define sets for time points and dimensions
        self.model.t = RangeSet(0, self.N - 1)
        self.model.dimensions = RangeSet(0, self.dimensions - 1)

        # define state variables over time and dimensions
        self.model.y = Var(self.model.t, self.model.dimensions, domain=pyo.Reals, 
                           initialize=0.1, bounds=(lower_bound, upper_bound))

        # slack variables for initial conditions
        self.model.slack = Var(self.model.dimensions, domain=pyo.Reals, 
                               bounds=(-1e-1, 1e-1), initialize=0.0)

        # initial condition constraints
        def init_condition_rule(m, d):
            return m.y[0, d] == self.initial_state[d] + m.slack[d]
        
        self.model.init_condition = Constraint(self.model.dimensions, rule=init_condition_rule)
        
        # objective function incorporating ODE penalties
        def _objective(m):
            penalty = 0
            
            # ODE penalties
            for i in m.t:
                # build neural network input
                nn_input = [m.y[i, d] for d in m.dimensions]

                # add time and extra inputs
                if not self.time_invariant:
                    nn_input.append(self.t[i])

                if self.extra_input is not None:
                    for input_array in self.extra_input.T:
                        nn_input.append(input_array[i])
                
                nn_output = self.nn_output(nn_input) 

                # accumulate penalties for each dimension
                for d in m.dimensions:
                    # dy_dt = sum(self.D[i-1, j-1] * m.y[j, d] for j in m.t)
                    dy_dt = sum(self.D[i, j] * m.y[j, d] for j in m.t)
                    penalty += (nn_output[d] - dy_dt)**2
            
            # penalty for slack variables
            slack_penalty = sum(m.slack[d]**2 for d in m.dimensions)
            return penalty + 1e6 * slack_penalty  # multiplied by a large constant to enforce initial conditions
        
        self.model.obj = Objective(rule=_objective, sense=pyo.minimize)
        
    def nn_output(self, nn_input):
        epsilon = 1e-10

        if len(self.layer_sizes) == 3:
            hidden = np.dot(self.W1, nn_input) + self.b1  # Shape: (hidden_layer_size,)
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
            else:
                raise ValueError("Unsupported activation function.")
                
            outputs = np.dot(self.W2, hidden) + self.b2  # Shape: (dimensions,)
        else:
            raise ValueError("Only 2-layer networks are supported.")
        
        return outputs  # Should be an array of length 'dimensions'

    def solve_model(self):
        # solve the model using IPOPT
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(self.model, tee=True)

        # extract solver information
        solver_info = {
            'solver_time': result.solver.time,
            'termination_condition': result.solver.termination_condition,
            'message': result.solver.message
        }
        
        print(solver_info)
        return solver_info
    
    def extract_solution(self):
        # Extract the solution for all time points and dimensions
        y_values = np.array([[value(self.model.y[i, d]) for d in self.model.dimensions] for i in self.model.t])
        return y_values  # Shape: (N, dimensions)
