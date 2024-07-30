import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet
from pyomo.dae import ContinuousSet, DerivativeVar

# To do:
# Rewrite this solver to work with discrete t points and using D matrix

class DirectODESolver:
    def __init__(self, t, layer_sizes, weights, biases, initial_state, D, 
                 act_func="tanh", time_invariant=True, extra_input=None,
                 params = None):
        
        self.t = t
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases
        self.initial_state = initial_state  
        self.act_func = act_func
        self.time_invariant = time_invariant
        self.extra_input = extra_input
        self.model = ConcreteModel()
        
        # model weights
        self.W1 = weights[0]
        self.W2 = weights[1]
        self.b1 = biases[0]
        self.b2 = biases[1]
        
        # first_derivative_matrix
        self.D = D
        
        self.model = ConcreteModel()
        
        self.params = params
        
    def build_model(self):
        self.N = len(self.t)
        
        lower_bound = -2.0
        upper_bound = 2.0
        
        self.model.t = RangeSet(0, self.N - 1)
        self.model.y = Var(self.model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        # self.model.init_condition = Constraint(expr= (self.model.y[0] == self.initial_state))
        
        self.model.ode = ConstraintList()
        
        for i in range(self.N):
            nn_input = [self.model.y[i]]

            # add time and extra inputs
            if not self.time_invariant:
                nn_input.append(self.t[i])

            if self.extra_input is not None:
                for input in self.extra_input.T:
                    nn_input.append(input[i])
            
            nn_y = self.nn_output(nn_input)
            dy_dt = sum(self.D[i, j] * self.model.y[j] for j in range(self.N))
            self.model.ode.add(nn_y == dy_dt)
        
        def _objective(m):
            # return np.abs(m.y[self.t[0]] - self.initial_state) 
            return 1
        
        self.model.obj = Objective(rule=_objective, sense=pyo.minimize)
        
    def nn_output(self, nn_input):

        if len(self.layer_sizes) == 3:
            hidden = np.dot(self.W1, nn_input) + self.b1
            epsilon = 1e-10
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
                
            outputs = np.dot(self.W2, hidden) + self.b2
        else:
            raise ValueError("Only 2 hidden layers are supported")
        
        return outputs

    def solve_model(self):
        # solve the model
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(self.model, tee=True)

        # extract solver information
        solver_time = result.solver.time
        termination_condition = result.solver.termination_condition
        message = result.solver.message

        solver_info = {
            'solver_time': solver_time,
            'termination_condition': termination_condition,
            'message': message
        }
        
        print(solver_info)
        return solver_info
    
    def extract_solution(self):
        y_values = np.array([value(self.model.y[i]) for i in range(self.N)])
        return y_values


