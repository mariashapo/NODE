import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet
from pyomo.dae import ContinuousSet, DerivativeVar

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax as dfx

class NeuralODEPyomoADMM:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, 
                 penalty_lambda_reg=0.1, penalty_lambda_smooth = 0.1,
                 discretization_scheme = "LAGRANGE-RADAU", ncp = 3,
                 act_func="tanh", w_init_method="random", params=None, y_init=None, 
                 deriv_method="collocation", is_continuous=True):
        
        """
        deriv_method = "collocation" or "pyomo"
        """
        
        midpoint = len(t)//2 
        self.y_observed1 = y_observed[:midpoint]
        self.y_observed2 = y_observed[midpoint:]
        self.t1 = t[ :midpoint ]
        self.t2 = t[ midpoint: ]
        self.model1 = ConcreteModel()
        self.model2 = ConcreteModel()
        self.D1 = first_derivative_matrix[0]
        self.D2 = first_derivative_matrix[1]
            
        if extra_input is not None:
            self.extra_input1 = extra_input[:midpoint]
            self.extra_input2 = extra_input[midpoint:]
        else:
            self.extra_input1 = None
            self.extra_input2 = None
        
        self.w_init_method = w_init_method
        self.act_func = act_func
        
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda_reg = penalty_lambda_reg
        self.layer_sizes = layer_sizes
        self.y_init = y_init
        self.time_invariant = time_invariant
        self.params = params
        self.observed_dim = None
        self.data_dim = None
        self.deriv_method = deriv_method
        self.is_continuous = is_continuous
        self.discretization_scheme = discretization_scheme
        self.penalty_lambda_smooth = penalty_lambda_smooth
        self.ncp = ncp 
        
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
        # M = number of observed variables (e.g., 2 for u and v)
        N, M = self.y_observed1.shape
        self.data_dim = N
        self.observed_dim = M

        self.model1.t = ContinuousSet(initialize=self.t1)
        self.model2.t = ContinuousSet(initialize=self.t2)

        lower_bound = -5.0
        upper_bound = 5.0

        self.model1.y = Var(self.model1.t, domain=pyo.Reals,
                             initialize=lambda m, t: self.y_observed1[(np.abs(self.t1 - t)).argmin()],
                             bounds=(lower_bound, upper_bound))

        self.model2.y = Var(self.model2.t, domain=pyo.Reals,
                             initialize=lambda m, t: self.y_observed2[(np.abs(self.t2 - t)).argmin()],
                             bounds=(lower_bound, upper_bound))

        weight_bounds = (-100.0, 100.0)
        input_size, layer1, output_size = self.layer_sizes[0], self.layer_sizes[1], self.layer_sizes[2]
        layer1 = self.layer_sizes[1]
        
        # First Set of Weights
        self.model1.W1 = Var(range(layer1), range(input_size), 
                            initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model1.b1 = Var(range(layer1), 
                            initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model1.W2 = Var(range(output_size), range(layer1), 
                            initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model1.b2 = Var(range(output_size), 
                            initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

        # Second Set of Weights
        self.model2.W1 = Var(range(layer1), range(input_size), 
                            initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model2.b1 = Var(range(layer1), 
                            initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model2.W2 = Var(range(output_size), range(layer1), 
                            initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model2.b2 = Var(range(output_size), 
                            initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

        # ----------------------------------------- CONSTRAINTS -------------------------------------------
        # ------------------------------------ DERIVATIVE VARIABLES ---------------------------------------
        self.model1.dy_dt = DerivativeVar(self.model1.y, wrt=self.model1.t)
        self.model2.dy_dt = DerivativeVar(self.model2.y, wrt=self.model2.t)
                
        def _con1_y(m, t_i):
            if t_i == self.t1[0]:
                return Constraint.Skip 
            
            # define the correct neural net input
            nn_input = [m.y[t_i]]  
            if not self.time_invariant:
                nn_input.append(t_i)
            if self.extra_input1 is not None:
                # determine the index of the extra inputs
                index = (np.abs(self.t1 - t_i)).argmin()
                for input in self.extra_input1.T:
                    nn_input.append(input[index])
            nn_y = self.nn_output(nn_input, m)
            return nn_y == m.dy_dt[t_i] 
        
        def _con2_y(m, t_i):
            if t_i == self.t2[0]:
                return Constraint.Skip 
            # define the correct neural net input
            nn_input = [m.y[t_i]]  
            if not self.time_invariant:
                nn_input.append(t_i)
            if self.extra_input2 is not None:
                # determine the index of the extra inputs
                index = (np.abs(self.t2 - t_i)).argmin()
                for input in self.extra_input2.T:
                    nn_input.append(input[index])
            nn_y = self.nn_output(nn_input, m)
            return nn_y == m.dy_dt[t_i] 
        
        self.model1.con1_y = Constraint(self.model1.t, rule=_con1_y)
        self.model2.con2_y = Constraint(self.model2.t, rule=_con2_y)
        
        # -------------------------------------- OBJECTIVE FUNCTIONS ----------------------------------------
        def _objective1(m):
            data_fit = sum((m.y[t_i] - self.y_observed1[i])**2 for i, t_i in enumerate(self.t1))
            reg_smooth = sum((m.y[self.t1[i+1]] - m.y[self.t1[i]])**2 for i in range(len(self.t1) - 1))

            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + 
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + 
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + 
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))
            
            return data_fit + reg * self.penalty_lambda_reg + reg_smooth * self.penalty_lambda_smooth

        def _objective2(m):
            data_fit = sum((m.y[t_i] - self.y_observed2[i])**2 for i, t_i in enumerate(self.t2))
            reg_smooth = sum((m.y[self.t2[i+1]] - m.y[self.t2[i]])**2 for i in range(len(self.t2) - 1))

            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + 
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + 
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + 
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

            return data_fit + reg * self.penalty_lambda_reg + reg_smooth * self.penalty_lambda_smooth
        
        self.model1.obj = Objective(rule=_objective1, sense=pyo.minimize)
        self.model2.obj = Objective(rule=_objective2, sense=pyo.minimize)
        
        # END BUILD MODEL
    
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
        # apply discretization
        if self.discretization_scheme == 'LAGRANGE-RADAU':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model1, nfe=len(self.t1)-1, ncp=self.ncp, scheme='LAGRANGE-RADAU')
            discretizer.apply_to(self.model2, nfe=len(self.t2)-1, ncp=self.ncp, scheme='LAGRANGE-RADAU')
        elif self.discretization_scheme == 'BACKWARD':
            discretizer = pyo.TransformationFactory('dae.finite_difference')
            discretizer.apply_to(self.model1, nfe=len(self.t1)-1, scheme='BACKWARD')
            discretizer.apply_to(self.model2, nfe=len(self.t2)-1, scheme='BACKWARD')
        elif self.discretization_scheme == 'LAGRANGE-LEGENDRE':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model1, nfe=len(self.t1)-1, ncp=self.ncp, scheme='LAGRANGE-LEGENDRE')
            discretizer.apply_to(self.model2, nfe=len(self.t2)-1, ncp=self.ncp, scheme='LAGRANGE-LEGENDRE')

        # solver: batch 2  
        solver = SolverFactory('ipopt')
        
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
                
        result = solver.solve(self.model2, tee=True)
        
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
        
        solver = SolverFactory('ipopt')
        
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        
        # solver: batch 1
        result = solver.solve(self.model1, tee=True)
        
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
        y1 = np.array([value(self.model1.y[self.t1[i]]) for i in range(self.data_dim)])
        y2 = np.array([value(self.model2.y[self.t2[i]]) for i in range(self.data_dim)])
        return [y1, y2]
   
    def extract_derivative(self):
        dy1_dt, dy2_dt = [], []
        for t in self.model1.t:
            try:
                dy1_dt.append((t, pyo.value(self.model1.dy_dt[t])))
            except ValueError:
                dy1_dt.append((t, None)) 
        
        for t in self.model2.t:
            try:
                dy2_dt.append((t, pyo.value(self.model2.dy_dt[t])))
            except ValueError:
                dy2_dt.append((t, None))                 
        
        return dy1_dt

    def extract_weights(self):
        weights_li = []
        
        for model_index, model in enumerate([self.model1, self.model2]):
            weights = {}
            W1 = np.array([[value(model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
            b1 = np.array([value(model.b1[j]) for j in range(self.layer_sizes[1])])
            W2 = np.array([[value(model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
            b2 = np.array([value(model.b2[j]) for j in range(self.layer_sizes[2])])
            weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
            print(f"Model index is {model_index}")
            weights_li.append(weights)
        # weights [model1_weights, model2_weights]
        return weights_li


    def predict(self, input_vector):
        weights_list = self.extract_weights()
        print(f"Number of weight sets: {len(weights_list)}")
        outputs = []
        
        # Compute output for each weight set
        for weights in weights_list:
            output = self.predict_single_set(weights, input_vector)
            outputs.append(output[0].item())  # Assuming output needs to be scalar per set

        return outputs
    
    def predict_single_set(self, weights, input_vector):
        """Compute the output of the neural network for a single set of weights."""
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
        hidden = jnp.tanh(jnp.dot(W1, input_vector) + b1)
        output = jnp.dot(W2, hidden) + b2
        return output

    def neural_ode(self, y0, t, extra_args=None): 
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)

            if args is not None:
                extra_inputs, t_all = args
                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    if extra_inputs.ndim == 2:
                        # use interpolation to obtain the value of the extra inputs at the time point t
                        interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(t, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, extra_inputs)

            result = self.predict(input)
            return result
        
        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            term, # function
            solver,
            t0=t[0],
            t1=t[-1],
            dt0 = 1e-3,
            y0=y0,
            args=extra_args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
        )
        #print("solution.ts", solution.ts)
        return solution.ys