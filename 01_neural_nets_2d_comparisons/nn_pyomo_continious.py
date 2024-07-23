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
                 penalty_lambda_reg=0.1, penalty_lambda_input=0.001, constraint_penalty=0.1, 
                 act_func="tanh", w_init_method="random", params=None, y_init=None, 
                 constraint="l1", deriv_method="collocation", is_continuous=True):
        
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
        self.penalty_lambda_input = penalty_lambda_input
        self.constraint_penalty = constraint_penalty
        self.constraint = constraint
        self.deriv_method = deriv_method
        self.is_continuous = is_continuous
        
    def update_y_observed(self, new_y_observed):
        """
        Update the y_observed attribute with new values.
        
        Parameters:
        new_y_observed (numpy.ndarray): The new observed data to update.
        """
        if not isinstance(new_y_observed, np.ndarray):
            raise ValueError("new_y_observed must be a numpy array.")
        if new_y_observed.shape != self.y_observed.shape:
            raise ValueError("new_y_observed must have the same shape as the original y_observed.")
        
        self.y_observed = new_y_observed
        self.y_init = new_y_observed
        
        print("y_observed has been updated.")
        
    def update_model_y(self, new_y):
        """
        Update the model.y variables with new values.
        
        Parameters:
        new_y (numpy.ndarray): The new values to update the model.y variables.
        """
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

        if self.is_continuous:
            model.t = ContinuousSet(initialize=self.t)
        else:
            model.t_idx = RangeSet(0, N - 1)
            model.var_idx = RangeSet(0, M - 1)

        # model.var_idx = RangeSet(0, M - 1)
        lower_bound = -5.0
        upper_bound = 5.0

        if self.y_init is None:
            if M == 1:
                model.y = Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
            elif M == 2:
                model.y1 = Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
                model.y2 = Var(model.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        else:
            if M == 1:
                model.y = Var(model.t, domain=pyo.Reals, 
                            initialize=lambda m, t: self.y_init[(np.abs(self.t - t)).argmin()], 
                            # y_init passed in would be a numpy array
                            # hence, need to determine the index of the closest time point
                            bounds=(lower_bound, upper_bound))
            elif M == 2:
                model.y1 = Var(model.t, domain=pyo.Reals, 
                            initialize=lambda m, t: self.y_init[0][(np.abs(self.t - t)).argmin()], 
                            bounds=(lower_bound, upper_bound))
                model.y2 = Var(model.t, domain=pyo.Reals, 
                            initialize=lambda m, t: self.y_init[1][(np.abs(self.t - t)).argmin()], 
                            bounds=(lower_bound, upper_bound))


        weight_bounds = (-100.0, 100.0)
        input_size = self.layer_sizes[0]
        layer1 = self.layer_sizes[1]
        
        model.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        model.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        
        if len(self.layer_sizes) == 3:
            output_size = self.layer_sizes[2]
            model.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
            model.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
            
        elif len(self.layer_sizes) == 4:
            layer2 = self.layer_sizes[2]
            output_size = self.layer_sizes[3]
            model.W2 = Var(range(layer2), range(layer1), initialize=0.1)
            model.b2 = Var(range(layer2), initialize=0.1)
            model.W3 = Var(range(output_size), range(layer2), initialize=0.1)
            model.b3 = Var(range(output_size), initialize=0.1)
        else:
            raise ValueError("layer_sizes should have exactly 3 elements: [input_size, hidden_size, output_size].")
        
        # ------------------------------------ DERIVATIVE VARIABLES ---------------------------------------
        if self.deriv_method == "pyomo":
            if M == 1:
                model.dy_dt = DerivativeVar(model.y, wrt=model.t)
                
                def _con1_y(m, t_i):
                    if t_i == self.t[0]:
                        return Constraint.Skip 
                    
                    # define the correct neural net input
                    nn_input = [model.y[t_i]]  
                    if not self.time_invariant:
                        nn_input.append(t_i)
                
                    if self.extra_inputs is not None:
                        index = (np.abs(self.t - t_i)).argmin()
                        for input in self.extra_inputs.T:
                            nn_input.append(input[index])

                    nn_y1 = self.nn_output(nn_input, m)
                    
                    return nn_y1 == model.dy_dt[t_i] 
                
                self.model.con1_y = Constraint(self.model.t, rule=_con1_y)
                
            elif M == 2: 
                model.dy1_dt = DerivativeVar(model.y1, wrt=model.t)
                model.dy2_dt = DerivativeVar(model.y2, wrt=model.t)

                def _con1_y1(m, t_i):
                    if t_i == self.t[0]:
                        return Constraint.Skip  
                    
                    # define the correct neural net input
                    nn_input = [model.y1[t_i], m.y2[t_i]]  
                    if not self.time_invariant:
                        nn_input.append(t_i)
                
                    if self.extra_inputs is not None:
                        index = (np.abs(self.t - t_i)).argmin()
                        for input in self.extra_inputs.T:
                            nn_input.append(input[index])

                    nn_y1, _ = self.nn_output(nn_input, m)

                    return nn_y1 == model.dy1_dt[t_i] 

                def _con1_y2(m, t_i):
                    if t_i == self.t[0]:
                        return Constraint.Skip 
                    
                    # define the correct neural net input
                    nn_input = [model.y1[t_i], m.y2[t_i]]  
                    if not self.time_invariant:
                        nn_input.append(t_i)
                
                    if self.extra_inputs is not None:
                        index = (np.abs(self.t - t_i)).argmin()
                        for input in self.extra_inputs.T:
                            nn_input.append(input[index])

                    _, nn_y2 = self.nn_output(nn_input, m)
                    
                    return nn_y2 == model.dy2_dt[t_i] 
                

                self.model.con1_y1 = Constraint(self.model.t, rule=_con1_y1)
                self.model.con1_y2 = Constraint(self.model.t, rule=_con1_y2)

        elif self.deriv_method == "collocation":
            model.ode = ConstraintList()
            # for each time point
            
            for i, t_i in enumerate(self.t):
                if i == 0:
                    continue  # Skip the first time point to avoid boundary issues

                if M == 1: 
                    # (np.abs(self.t - t)).argmin()
                    dy_dt = sum(self.first_derivative_matrix[i, j] * model.y[self.t[j]] for j in range(N))
                    nn_input = [model.y[t_i]]  
                elif M == 2:
                    dy1_dt = sum(self.first_derivative_matrix[i, j] * model.y1[self.t[j]] for j in range(N))
                    dy2_dt = sum(self.first_derivative_matrix[i, j] * model.y2[self.t[j]] for j in range(N))
                    nn_input = [model.y1[t_i], model.y2[t_i]]

                # add time and extra inputs
                if not self.time_invariant:
                    nn_input.append(self.t[i])
                
                if self.extra_inputs is not None:
                    # the expected shape is (N, num_extra_inputs)
                    for input in self.extra_inputs.T:
                        nn_input.append(input[i])

                if M == 1:
                    nn_y = self.nn_output(nn_input, model)
                    if self.constraint == "l2":
                        model.ode.add((nn_y - dy_dt)**2 == 0)
                    elif self.constraint == "l1":
                        model.ode.add(nn_y == model.dy_dt[t_i])
                    else:
                        raise ValueError("Constraint should be either 'l1' or 'l2'.")
                    
                elif M == 2:
                    nn_y1, nn_y2 = self.nn_output(nn_input, model)
                    
                    if self.constraint == "l2":
                        model.ode.add((nn_y1 - dy1_dt)**2 == 0)
                        model.ode.add((nn_y2 - dy2_dt)**2 == 0)
                    elif self.constraint == "l1":
                        model.ode.add((nn_y1 == dy1_dt))
                        model.ode.add((nn_y2 == dy2_dt))
                    else:
                        raise ValueError("Constraint should be either 'l1' or 'l2'.")
        else:
            raise ValueError("deriv_method should be either 'collocation' or 'pyomo'.")
    
        def _objective(m):
            if M == 1:
                data_fit = sum((m.y[t_i] - self.y_observed[i])**2 for i, t_i in enumerate(self.t))
                # penalty = sum((m.y[t_i] - self.y_init[t_i])**2 for t_i in self.t) if self.y_init is not None else 0
            elif M == 2:
                data_fit = sum((m.y1[t_i] - self.y_observed[i, 0])**2 + (m.y2[t_i] - self.y_observed[i, 1])**2 for i, t_i in enumerate(self.t))
                # enalty = sum((m.y1[t_i] - self.y_init[0][t_i])**2 + (m.y2[t_i] - self.y_init[1][t_i])**2 for t_i in self.t) if self.y_init is not None else 0    
                
            reg = sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) + \
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) + \
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) + \
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2]))
            
            return data_fit + reg * self.penalty_lambda_reg # + self.penalty_lambda_input * penalty**2

        model.obj = Objective(rule=_objective, sense=pyo.minimize)
        self.model = model
    
    def nn_output(self, nn_input, m):

        if len(self.layer_sizes) == 3:
            hidden = np.dot(m.W1, nn_input) + m.b1
            epsilon = 1e-10
            if self.act_func == "tanh":
                hidden = [pyo.tanh(h) for h in hidden]
            elif self.act_func == "sigmoid":
                hidden = [1 / (1 + pyo.exp(-h) + epsilon) for h in hidden]
            elif self.act_func == "softplus":
                hidden = [pyo.log(1 + pyo.exp(h) + epsilon) for h in hidden]
                
            outputs = np.dot(m.W2, hidden) + m.b2
        
        elif len(self.layer_sizes) == 4:
            hidden1 = np.dot(m.W1, nn_input) + m.b1
            
            if self.act_func == "tanh":
                hidden1 = [pyo.tanh(h) for h in hidden1]
            elif self.act_func == "sigmoid":
                hidden1 = [1 / (1 + pyo.exp(-h)) for h in hidden1]
            elif self.act_func == "softplus":
                hidden1 = [pyo.log(1 + pyo.exp(h)) for h in hidden1]
            
            hidden2 = np.dot(m.W2, hidden1) + m.b2
            
            if self.act_func == "tanh":
                hidden2 = [pyo.tanh(h) for h in hidden2]
            elif self.act_func == "sigmoid":
                hidden2 = [1 / (1 + pyo.exp(-h)) for h in hidden2]
            elif self.act_func == "softplus":
                hidden2 = [pyo.log(1 + pyo.exp(h)) for h in hidden2]
            
            outputs = np.dot(m.W3, hidden2) + m.b3
        else:
            raise ValueError("layer_sizes should have exactly 3 elements: [input_size, hidden_size, output_size].")
        
        return outputs

    def solve_model(self):
        # Apply discretization
        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(self.model, nfe=len(self.t)-1, ncp=3, scheme='LAGRANGE-RADAU')
        
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
            dy_dt = np.array([pyo.value(self.model.dy_dt[t]) for t in self.model.t])
            return dy_dt
        elif self.observed_dim == 2:
            dy1_dt = np.array([pyo.value(self.model.dy1_dt[t]) for t in self.model.t])
            dy2_dt = np.array([pyo.value(self.model.dy2_dt[t]) for t in self.model.t])
            return [dy1_dt, dy2_dt]


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
            # dt0= t[-1] - t[0],
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
