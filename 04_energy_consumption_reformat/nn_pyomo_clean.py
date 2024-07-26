import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet
from pyomo.dae import ContinuousSet, DerivativeVar

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax as dfx

import warnings

class NeuralODEPyomo:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, 
                 penalty_lambda_reg=0.1, penalty_lambda_smooth = 0.1,
                 discretization_scheme = "LAGRANGE-RADAU", ncp = 3,
                 act_func="tanh", w_init_method="random", params=None, y_init=None, 
                deriv_method="collocation", is_continuous=True):
        
        """
        deriv_method = "collocation" or "pyomo"
        """
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda_reg = penalty_lambda_reg
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.model = ConcreteModel()
        self.y_init = y_init
        self.w_init_method = w_init_method
        self.time_invariant = time_invariant
        self.extra_inputs = extra_input
        self.params = params
        self.observed_dim = None
        self.data_dim = None
        self.deriv_method = deriv_method
        self.is_continuous = is_continuous
        self.discretization_scheme = discretization_scheme
        self.penalty_lambda_smooth = penalty_lambda_smooth
        self.ncp = ncp
        
    def update_y_observed(self, new_y_observed):
        if not isinstance(new_y_observed, np.ndarray):
            raise ValueError("new_y_observed must be a numpy array.")
        if new_y_observed.shape != self.y_observed.shape:
            raise ValueError("new_y_observed must have the same shape as the original y_observed.")
        
        self.y_observed = new_y_observed
        self.y_init = new_y_observed
        
        print("y_observed has been updated.")
        
    def update_model_y(self, new_y):
        if not isinstance(new_y, np.ndarray):
            raise ValueError("new_y must be a numpy array.")
        if self.observed_dim == 1:
            if new_y.shape[0] != self.data_dim:
                raise ValueError("new_y must have the same number of time steps as the original data.")
            for i in range(self.data_dim):
                self.model.y[i].set_value(new_y[i])
        elif self.observed_dim == 2:
            if new_y.shape != (self.data_dim, 2):
                raise ValueError("new_y must have the same shape as the original data (N, 2).")
            for i in range(self.data_dim):
                self.model.y1[i].set_value(new_y[i, 0])
                self.model.y2[i].set_value(new_y[i, 1])
        print("model.y has been updated.")    
        
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
        N, M = self.y_observed.shape
        self.data_dim = N
        self.observed_dim = M
        model = self.model
        model.t = ContinuousSet(initialize=self.t)

        lower_bound = -5.0
        upper_bound = 5.0

        if self.y_init is None:
            model.y = Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        else:
            if M == 1:
                model.y = Var(model.t, domain=pyo.Reals, 
                            initialize=lambda m, t: self.y_init[(np.abs(self.t - t)).argmin()], 
                            bounds=(lower_bound, upper_bound))

        weight_bounds = (-100.0, 100.0)
        input_size = self.layer_sizes[0]
        layer1 = self.layer_sizes[1]
        
        model.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        model.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        
        output_size = self.layer_sizes[2]
        model.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        model.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
    

        # ------------------------------------ COLLOCATION ---------------------------------------
        model.ode = ConstraintList()
        
        for i, t_i in enumerate(self.t):
            if i == 0:
                continue 

            dy_dt = sum(self.first_derivative_matrix[i, j] * model.y[self.t[j]] for j in range(N))
            nn_input = [model.y[t_i]]  

            # add time and extra inputs
            if not self.time_invariant:
                nn_input.append(self.t[i])
            
            if self.extra_inputs is not None:
                # the expected shape is (N, num_extra_inputs)
                for input in self.extra_inputs.T:
                    nn_input.append(input[i])
                
            nn_y = self.nn_output(nn_input, model)
            model.ode.add(nn_y == dy_dt)
                
        
        def _objective(m):
            data_fit = sum((m.y[t_i] - self.y_observed[i])**2 for i, t_i in enumerate(self.t))

            # regularization for weights and biases
            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + 
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + 
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + 
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

            return data_fit + reg * self.penalty_lambda_reg #+ reg_smooth_derivative

        model.obj = Objective(rule=_objective, sense=pyo.minimize)
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
        # Apply discretization
        if self.discretization_scheme == 'LAGRANGE-RADAU':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, ncp=self.ncp, scheme='LAGRANGE-RADAU')
        elif self.discretization_scheme == 'BACKWARD':
            discretizer = pyo.TransformationFactory('dae.finite_difference')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, scheme='BACKWARD')
        elif self.discretization_scheme == 'LAGRANGE-LEGENDRE':
            discretizer = pyo.TransformationFactory('dae.collocation')
            discretizer.apply_to(self.model, nfe=len(self.t)-1, ncp=self.ncp, scheme='LAGRANGE-LEGENDRE')

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
        if self.observed_dim == 1:
            y = np.array([value(self.model.y[self.t[i]]) for i in range(self.data_dim)])
            return y
        elif self.observed_dim == 2:
            y1 = np.array([value(self.model.y1[self.t[i]]) for i in range(self.data_dim)])
            y2 = np.array([value(self.model.y2[self.t[i]]) for i in range(self.data_dim)])
            return [y1, y2]
    
    def extract_y_observed(self):
       return self.y_observed
   
    def extract_derivative(self):
        if self.observed_dim == 1:
            dy_dt = []
            for i in range(self.data_dim):
                dy_dt_i = sum(self.first_derivative_matrix[i, j] * pyo.value(self.model.y[t_i]) for j, t_i in enumerate(self.t))
                dy_dt.append(dy_dt_i)
            return np.array(dy_dt)
        elif self.observed_dim == 2:
            dy1_dt = []
            dy2_dt = []
            for i in range(self.data_dim):
                t_i = self.t[i]
                dy1_dt_i = sum(self.first_derivative_matrix[i, j] * pyo.value(self.model.y1[self.model.t[j]]) for j in range(self.data_dim))
                dy2_dt_i = sum(self.first_derivative_matrix[i, j] * pyo.value(self.model.y2[self.model.t[j]]) for j in range(self.data_dim))
                dy1_dt.append(dy1_dt_i)
                dy2_dt.append(dy2_dt_i)
            return [np.array(dy1_dt), np.array(dy2_dt)]

    def extract_weights(self):
        weights = {}
        
        W1 = np.array([[value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
        weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
        
        if len(self.layer_sizes) == 4:
            W3 = np.array([[value(self.model.W3[j, k]) for k in range(self.layer_sizes[2])] for j in range(self.layer_sizes[3])])
            b3 = np.array([value(self.model.b3[j]) for j in range(self.layer_sizes[3])])
            weights['W3'], weights['b3'] = W3, b3
            
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

    def mae(self, y_true, y_pred):
        mae_result = np.mean(np.abs(y_true - y_pred))
        return mae_result
    
    def mse(self, y_true, y_pred):
        mse_result = np.mean(np.squared(y_true - y_pred))
        return mse_result
    
    def neural_ode_old(self, y0, t, extra_args = None):
        
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
            
            result = self.predict(input)
            return result
    
        return odeint(func, y0, t, extra_args)

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
                        interpolated_inputs = jnp.array([linear_interpolate(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = linear_interpolate(t, t_all, extra_inputs)
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
      
def linear_interpolate(x, xp, yp):
    """
    Interpolate data points (xp, yp) to find the value at x.
    """
    return jnp.interp(x, xp, yp)
