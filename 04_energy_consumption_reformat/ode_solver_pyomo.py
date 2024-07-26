import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet
from pyomo.dae import ContinuousSet, DerivativeVar

class DirectODESolver:
    def __init__(self, t, layer_sizes, weights, biases, initial_state, params = None, act_func="tanh", time_invariant=True, extra_input=None,  discretization_scheme = 'LAGRANGE-RADAU'):
        self.t = t
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases
        self.initial_state = initial_state  
        self.act_func = act_func
        self.time_invariant = time_invariant
        self.extra_inputs = extra_input
        self.model = ConcreteModel()
        self.W1 = weights[0]
        self.W2 = weights[1]
        self.b1 = biases[0]
        self.b2 = biases[1]
        self.params = params
        self.discretization_scheme = discretization_scheme
        
        """
        W1 = trained_weights_biases['W1']
        b1 = trained_weights_biases['b1']
        W2 = trained_weights_biases['W2']
        b2 = trained_weights_biases['b2']
        """
        
    def build_model(self):
        self.model.t = ContinuousSet(initialize=self.t)
        self.model.y = Var(self.model.t, domain=pyo.Reals, initialize=1)
        self.model.dy_dt = DerivativeVar(self.model.y, wrt=self.model.t)
        
        # self.model.init_condition = Constraint(expr= (self.model.y[self.t[0]] == self.initial_state ))
        
        def _ode(m, t_i):
            nn_input = [m.y[t_i]]
            if not self.time_invariant:
                nn_input.append(t_i)
            if self.extra_inputs is not None:
                index = (np.abs(self.t - t_i)).argmin()
                for input in self.extra_inputs.T:
                    nn_input.append(input[index])
            nn_output = self.nn_output(nn_input)
            return nn_output == m.dy_dt[t_i]
        
        self.model.ode = Constraint(self.model.t, rule=_ode)
        
        def _objective(m):
            return np.abs(m.y[self.t[0]] - self.initial_state)
            #return 1
        
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
        # Apply discretization
        if self.discretization_scheme == 'LAGRANGE-RADAU':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, ncp=1, scheme='LAGRANGE-RADAU')
        elif self.discretization_scheme == 'BACKWARD':
            discretizer = pyo.TransformationFactory('dae.finite_difference')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, scheme='BACKWARD')
        elif self.discretization_scheme == 'LAGRANGE-LEGENDRE':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, ncp=3, scheme='LAGRANGE-LEGENDRE')

        # Solve the model
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(self.model, tee=True)

        # Extract solver information
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
        t_values = np.array([t for t in self.model.t])
        y_values = np.array([value(self.model.y[t]) for t in self.model.t])
        return t_values, y_values


