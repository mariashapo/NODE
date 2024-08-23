import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet

# To do:
# Rewrite this solver to work with discrete t points and using D matrix

class DirectODESolver:
    def __init__(self, t, layer_sizes, trained_weights_biases, initial_state, D, 
                 act_func="tanh", time_invariant=True, extra_input=None,
                 params = None):
        
        self.t = t
        self.layer_sizes = layer_sizes
        self.initial_state = initial_state  
        self.act_func = act_func
        self.time_invariant = time_invariant
        self.extra_input = extra_input
        self.model = ConcreteModel()
        
        # model weights
        self.W1 = trained_weights_biases['W1']
        self.W2 = trained_weights_biases['W2']
        self.b1 = trained_weights_biases['b1']
        self.b2 = trained_weights_biases['b2']
        
        # first_derivative_matrix
        self.D = D
        self.model = ConcreteModel()
        
        self.params = params
        
    def build_model(self):
        self.N = len(self.t)
        
        lower_bound = -5.0
        upper_bound = 5.0
        
        self.model.t = RangeSet(0, self.N - 1)
        self.model.y = Var(self.model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))

        self.model.slack = Var(domain=pyo.Reals, bounds=(-1e-1, 1e-1), initialize=0.0) 
        self.model.init_condition = Constraint(expr=(self.model.y[0] == self.initial_state + self.model.slack))

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
            return 1 + 1e6 * m.slack
        
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
        
        #print(solver_info)
        return solver_info
    
    def extract_solution(self):
        y_values = np.array([value(self.model.y[i]) for i in range(self.N)])
        return y_values
    
    def check_violated_constraints(self, tolerance=1e-8):
        violated_constraints = []
        initial_constr = self.model.init_condition
        constraint_list = self.model.ode

        if initial_constr.body() is not None:
            if not initial_constr.lower is None and value(initial_constr.body()) < value(initial_constr.lower) - tolerance:
                violated_constraints.append(("initial_condition", "lower", value(initial_constr.body()), value(initial_constr.lower)))
            if not initial_constr.upper is None and value(initial_constr.body()) > value(initial_constr.upper) + tolerance:
                violated_constraints.append(("initial_condition", "upper", value(initial_constr.body()), value(initial_constr.upper)))
        
        for i in range(1, len(constraint_list) + 1):
            constr = constraint_list[i]
            if constr.body() is not None:  # Ensure there is a body expression
                if not constr.lower is None and value(constr.body()) < value(constr.lower) - tolerance:
                    violated_constraints.append((i, "lower", value(constr.body()), value(constr.lower)))
                if not constr.upper is None and value(constr.body()) > value(constr.upper) + tolerance:
                    violated_constraints.append((i, "upper", value(constr.body()), value(constr.upper)))

        if violated_constraints:
            print("\nViolated Constraints:")
            for v_constr in violated_constraints:
                index, bound_type, body_val, bound_val = v_constr
                print(f"Constraint index: {index} - {bound_type} bound violated")
                print(f"Value: {body_val} vs. Bound: {bound_val}")
        else:
            print("No explicit violations found (note: IPOPT may still consider the problem infeasible due to numerical issues or other considerations).")


