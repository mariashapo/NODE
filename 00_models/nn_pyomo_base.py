import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory, value, RangeSet

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from scipy.integrate import ode, solve_ivp
import diffrax as dfx
from diffrax import ImplicitEuler, ODETerm

import os
import pickle
import warnings

class NeuralODEPyomo:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, 
                 time_invariant = True, extra_input = None, 
                 penalty_lambda_reg=0.1,
                 act_func="tanh", w_init_method="random", 
                 params = None, y_init = None, constraint = "l1",
                 y_collocation = None):
        
        np.random.seed(42)
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
        self.params = params.copy()
        self.observed_dim = None
        self.data_dim = None
        self.constraint = constraint
        
        if y_collocation is not None:
            self.y_collocation = y_collocation
        else:
            self.y_collocation = y_observed

# ---------------------------------------------------- INITIALIZATION ----------------------------------------------------

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
        # N : number of time steps; M : number of observed variables 
        N, M = self.y_observed.shape 
        # rows, columns
        self.data_dim, self.observed_dim = N, M
        
        self.model.t_idx = RangeSet(0, N - 1)

        lower_bound = -5.0
        upper_bound = 5.0

        if self.y_init is None:
            if M == 1:
                self.model.y = Var(self.model.t_idx, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
            elif M == 2:
                self.model.y1 = pyo.Var(self.model.t_idx, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
                self.model.y2 = pyo.Var(self.model.t_idx, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        else:
            if self.y_init.shape[0] < self.y_init.shape[1]:
                # less rows than columns
                warnings.warn("y_init should be structured such that each row represents a new time point.")
            if M == 1:
                self.model.y = pyo.Var(self.model.t_idx, domain=pyo.Reals, initialize=lambda m, i: self.y_init[i], bounds=(lower_bound, upper_bound))
                print(self.model.y)
            elif M == 2:
                self.model.y1 = pyo.Var(self.model.t_idx, domain=pyo.Reals, initialize=np.array(self.y_init[0]), bounds=(lower_bound, upper_bound))
                self.model.y2 = pyo.Var(self.model.t_idx, domain=pyo.Reals, initialize=np.array(self.y_init[1]), bounds=(lower_bound, upper_bound))

        weight_bounds = (-100.0, 100.0)
        input_size = self.layer_sizes[0]
        layer1 = self.layer_sizes[1]
        
        self.model.W1 = pyo.Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model.b1 = pyo.Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        
        if len(self.layer_sizes) == 3:
            output_size = self.layer_sizes[2]
            self.model.W2 = pyo.Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
            self.model.b2 = pyo.Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
            
        elif len(self.layer_sizes) == 4:
            layer2 = self.layer_sizes[2]
            output_size = self.layer_sizes[3]
            self.model.W2 = pyo.Var(range(layer2), range(layer1), initialize=0.1)
            self.model.b2 = pyo.Var(range(layer2), initialize=0.1)
            self.model.W3 = pyo.Var(range(output_size), range(layer2), initialize=0.1)
            self.model.b3 = pyo.Var(range(output_size), initialize=0.1)
        else:
            raise ValueError("layer_sizes should have exactly 3 elements: [input_size, hidden_size, output_size].")
        
        # CONSTRAINTS
        if M == 1:
            self.model.init_condition = Constraint(expr=(self.model.y[0] == self.y_observed[0][0])) # + self.model.slack
        elif M == 2:
            self.model.init_condition1 = Constraint(expr=(self.model.y1[0] == self.y_observed[0][0]))
            self.model.init_condition2 = Constraint(expr=(self.model.y2[0] == self.y_observed[0][1]))
        
        self.model.ode = ConstraintList()
        
        # for each collocation data point
        for i in range(1, N):
            if M == 1:
                dy_dt = sum(self.first_derivative_matrix[i, j] * self.model.y[j] for j in range(N))
                nn_input = [self.model.y[i]]  
            elif M == 2:
                dy1_dt = sum(self.first_derivative_matrix[i, j] * self.model.y1[j] for j in range(N))
                dy2_dt = sum(self.first_derivative_matrix[i, j] * self.model.y2[j] for j in range(N))
                nn_input = [self.model.y1[i], self.model.y2[i]]
            
            # add time and extra inputs
            if not self.time_invariant:
                nn_input.append(self.t[i])
            
            if self.extra_inputs is not None:
                # the expected shape is (N, num_extra_inputs)
                for input in self.extra_inputs.T:
                    nn_input.append(input[i])
            
            if M == 1:
                nn_y = self.nn_output(nn_input, self.model)
                if self.constraint == "l2":
                    self.model.ode.add((nn_y - dy_dt)**2 == 0)
                elif self.constraint == "l1":
                    self.model.ode.add((nn_y == dy_dt))
                elif self.constraint == "l2_inequality":
                    self.model.ode.add((nn_y - dy_dt)**2 <= self.constraint_penalty)
                
            elif M == 2:
                # nn_u, nn_v = self.nn_output(nn_input, model)
                nn_y1, nn_y2 = self.nn_output(nn_input, self.model)
                
                if self.constraint == "l2":
                    self.model.ode.add((nn_y1 - dy1_dt)**2 == 0)
                    self.model.ode.add((nn_y2 - dy2_dt)**2 == 0)
                elif self.constraint == "l1":
                    self.model.ode.add((nn_y1 == dy1_dt))
                    self.model.ode.add((nn_y2 == dy2_dt))
    
        def _objective(m):
            if M == 1:
                # mse
                data_fit = sum((m.y[i] - self.y_observed[i])**2 for i in m.t_idx) / len(m.t_idx)
            elif M == 2:
                data_fit = sum((m.y1[i] - self.y_observed[i, 0])**2 + (m.y2[i] - self.y_observed[i, 1])**2 for i in m.t_idx) / len(m.t_idx)

            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) +
                sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) +
                sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) +
                sum(m.b2[j]**2 for j in range(self.layer_sizes[2]))) / \
                (self.layer_sizes[1]*self.layer_sizes[0] + self.layer_sizes[2]*self.layer_sizes[1] + self.layer_sizes[1] + self.layer_sizes[2])

            return data_fit + reg * self.penalty_lambda_reg 

        # print('_objective')
        self.model.obj = Objective(rule=_objective, sense=pyo.minimize)

# ---------------------------------------------------------- TRAINING -----------------------------------------------------

    def nn_output(self, nn_input, m):
        # print('nn_output')
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

    def initialize_solver(self):
        solver = pyo.SolverFactory('ipopt')
        
        if self.params is not None:
            for key, value in self.params.items():
                if key == "detailed_output" and value:
                    solver.options['output_file'] = 'ipopt_output.log'
                    continue
                solver.options[key] = value
                
        return solver
    
    def extract_solver_info(self, result):
        solver_time = result.solver.time
        termination_condition = result.solver.termination_condition
        message = result.solver.message
        
        solver_info = {
            'solver_time': solver_time,
            'termination_condition': termination_condition,
            'message': message
        }
        
        return solver_info
    
    def solve_model(self):
        solver = self.initialize_solver()
        
        result = solver.solve(self.model, tee=True)
        
        return self.extract_solver_info(result)

    def solve_model_checkpoints(self, iter_per_check=20):
        folder = 'checkpoints'
        NeuralODEPyomo.clear_folder(folder)
        
        if 'max_iter' in self.params.keys():
            total_max_iter = self.params['max_iter']
        else:
            total_max_iter = 3000
        
        print(f"Total max iterations: {total_max_iter}")
        
        self.params['max_iter'] = iter_per_check
        solver = self.initialize_solver()
        
        checkpoint_results = []
        
        iter = 0
        self.save_checkpoint(iter, folder)
        while True and iter < total_max_iter:
            if iter != 0:
                solver = self.initialize_solver()
                self.load_checkpoint(f"{folder}/checkpoint_epoch_{iter}.pkl")

            result = solver.solve(self.model, tee=True)
            iter += iter_per_check
            
            self.save_checkpoint(iter, folder)

            y_pred = self.neural_ode_odeint(self.y_observed[0], self.t, (self.extra_inputs, self.t))
            checkpoint_results.append(y_pred)
            
            if result.solver.termination_condition == pyo.TerminationCondition.optimal:
                break
            
        return self.extract_solver_info(result), checkpoint_results
    
    # ------------------------------------------------------- PREDICTIONS ----------------------------------------------------

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

# ----------------------------------------------------- EXTRACT RESULTS ----------------------------------------------------
    def extract_solution(self):
        if self.observed_dim == 1:
            y = np.array([pyo.value(self.model.y[i]) for i in self.model.t_idx])
            return y
        elif self.observed_dim == 2:
            y1 = np.array([pyo.value(self.model.y1[i]) for i in self.model.t_idx])
            y2 = np.array([pyo.value(self.model.y2[i]) for i in self.model.t_idx])
            return [y1, y2]

    def extract_weights(self):
        weights = {}
        
        W1 = np.array([[pyo.value(self.model.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([pyo.value(self.model.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[pyo.value(self.model.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([pyo.value(self.model.b2[j]) for j in range(self.layer_sizes[2])])
        weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2
        
        if len(self.layer_sizes) == 4:
            W3 = np.array([[pyo.value(self.model.W3[j, k]) for k in range(self.layer_sizes[2])] for j in range(self.layer_sizes[3])])
            b3 = np.array([pyo.value(self.model.b3[j]) for j in range(self.layer_sizes[3])])
            weights['W3'], weights['b3'] = W3, b3
            
        return weights


# --------------------------------------------------------- CHECKPOINTS ----------------------------------------------------
    def save_checkpoint(self, iter, folder="checkpoints"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, f"checkpoint_epoch_{iter}.pkl")
        model_state = {
            'vars': {var.name: {index: pyo.value(var[index]) for index in var} for var in self.model.component_objects(pyo.Var)},
            'epoch': iter
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    def load_checkpoint(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        for var_name, values in data['vars'].items():
            model_var = getattr(self.model, var_name)
            for index, value in values.items():
                model_var[index].set_value(value)
        return data['epoch']

    def load_weights(self, weights):
        """A method to load weights into pyomo model variables from a dictionary."""
        for w_name, w_values in weights.items():
            layer_var = getattr(self.model, w_name)
            for idx, val in np.ndenumerate(w_values):
                layer_var[idx].value = val

# --------------------------------------------------------- ODE DIRECT SOLVERS --------------------------------------------
    
    def func_interpolated(self, y, t, args):
        input = jnp.atleast_1d(y)
        if not self.time_invariant:
            input = jnp.append(input, t)

        if args is not None:
            extra_inputs, t_all = args
            if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                if extra_inputs.ndim == 2:
                    # Interpolation or index selection based on method needs
                    interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                    input = jnp.append(input, interpolated_inputs)
                elif extra_inputs.ndim == 1:
                    interpolated_input = jnp.interp(t, t_all, extra_inputs)
                    input = jnp.append(input, interpolated_input)
            else:
                input = jnp.append(input, extra_inputs)
        result = self.predict(input)
        return result
    
    def func_regular(self, y, t, args):
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
    
    def neural_ode_odeint(self, y0, t, extra_args = None, interpolated = True): 
        if interpolated:
            return odeint(self.func_interpolated, y0, t, extra_args)
        else:
            return odeint(self.func_regular, y0, t, extra_args)

    def neural_ode(self, y0, t, extra_args = None, interpolated = True, rtol = 1e-3, atol = 1e-6, dt0 = 1e-3): 
        
        if interpolated:
            term = dfx.ODETerm(lambda t, y, args: self.func_interpolated(y, t, args))
        else:
            term = dfx.ODETerm(lambda t, y, args: self.func_regular(y, t, args))
            
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            term, # function
            solver,
            t0=t[0],
            t1=t[-1],
            dt0 = dt0,
            y0=y0,
            args=extra_args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
        )
        #print("solution.ts", solution.ts)
        return solution.ys
    
    def neural_ode_vode(self, y0, t, extra_args=None, rtol=1e-3, atol=1e-6, dt0=1e-3, nsteps = 5000):
        def func(t, y):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)

            if extra_args is not None:
                extra_inputs, t_all = extra_args
                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    if extra_inputs.ndim == 2:
                        # Use interpolation to obtain the value of the extra inputs at the time point t
                        interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(t, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, extra_inputs)

            result = self.predict(input)
            return result

        print('reloaded')
        solver = ode(func).set_integrator('vode', method='bdf', rtol=rtol, atol=atol, nsteps=nsteps)
        solver.set_initial_value(y0, t[0])
        ts = [t[0]]
        ys = [y0]

        while solver.successful() and solver.t < t[-1]:
            solver.integrate(solver.t + dt0)
            ts.append(solver.t)
            ys.append(solver.y)

        return np.array(ts), np.array(ys)
        
    def neural_ode_euler(self, y0, t, extra_args=None):
            y = jnp.array(y0)
            ys = [y]
            
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                y = self.implicit_euler_step(y, t[i - 1], t[i], dt, extra_args)
                ys.append(y)
            
            return jnp.array(ys)

    def implicit_euler_step(self, y, current_time, next_time, dt, args):
        # Initial guess for y_next (can be the previous y, or more sophisticated guesses can be used)
        y_next = y
        
        def func(y_n):
            input = jnp.atleast_1d(y_n)
            if not self.time_invariant:
                input = jnp.append(input, next_time)

            if args is not None:
                extra_inputs, t_all = args
                if isinstance(extra_inputs, (jnp.ndarray, np.ndarray)):
                    if extra_inputs.ndim == 2:
                        interpolated_inputs = jnp.array([jnp.interp(next_time, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(next_time, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, args)

            return input

        # Function to be solved implicitly
        F = lambda y_n: y_n - y - dt * self.predict(func(y_n))

        # Using simple fixed-point iteration as an alternative to Newton-Raphson for demonstration
        for _ in range(10):  # Maximum 10 iterations
            y_next = y + dt * self.predict(func(y_next))
        
        return y_next

# ----------------------------------------------- HELPER DEBUGGING FUNCTIONS -----------------------------------------------

    def check_violated_constraints(self, tolerance=1e-8):
        violated_constraints = []
        constraint_list = self.model.ode

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
    
    @staticmethod
    def clear_folder(folder_path):
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(folder_path)
            print(f"Folder {folder_path} created")
