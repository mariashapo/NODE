import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, SolverFactory, value, RangeSet

import jax
import jax.numpy as jnp
import diffrax as dfx

np.random.seed(42)

class NeuralODEPyomoADMM:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, rho = 1.0,
                 penalty_lambda_reg=0.01, penalty_lambda_smooth=0.0, act_func="tanh", w_init_method="random", params=None, y_init=None):

        self.midpoint = len(t) // 2 
        self.y_observed1 = y_observed[:self.midpoint]
        self.y_observed2 = y_observed[self.midpoint:]
        self.t1 = t[:self.midpoint]
        self.t2 = t[self.midpoint:]
        self.model1 = ConcreteModel()
        self.model2 = ConcreteModel()

        if extra_input is not None:
            self.extra_input1 = extra_input[:self.midpoint]
            self.extra_input2 = extra_input[self.midpoint:]
        else:
            self.extra_input1 = None
            self.extra_input2 = None

        if y_init is not None:
            self.y_init1 = y_init[:self.midpoint]
            self.y_init2 = y_init[self.midpoint:]
        else:
            self.y_init1 = None
            self.y_init2 = None

        self.D1 = first_derivative_matrix[0]
        self.D2 = first_derivative_matrix[1]

        self.penalty_lambda_reg = penalty_lambda_reg
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.time_invariant = time_invariant
        self.params = params
        self.penalty_lambda_smooth = penalty_lambda_smooth
        
        self.rho = rho
        
        self.iter = 0

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

        self.model1.t = RangeSet(0, N - 1)
        self.model2.t = RangeSet(0, N - 1)

        lower_bound = -5.0
        upper_bound = 5.0

        if self.y_init1 is None:
            self.model1.y = Var(self.model1.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
            self.model2.y = Var(self.model2.t, domain=pyo.Reals, initialize=0.1, bounds=(lower_bound, upper_bound))
        else:
            self.model1.y = pyo.Var(self.model1.t, domain=pyo.Reals, initialize=lambda m, i: self.y_init1[i], bounds=(lower_bound, upper_bound))
            self.model2.y = pyo.Var(self.model2.t, domain=pyo.Reals, initialize=lambda m, i: self.y_init2[i], bounds=(lower_bound, upper_bound))

        weight_bounds = (-100.0, 100.0)
        input_size, layer1, output_size = self.layer_sizes[0], self.layer_sizes[1], self.layer_sizes[2]

        # model 1 weights and biases
        self.model1.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model1.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model1.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model1.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

        # model 2 weights and biases        
        self.model2.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model2.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model2.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model2.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
        
        # consensus
        self.W1_consensus = np.zeros((layer1, input_size))
        self.b1_consensus = np.zeros(layer1)
        self.W2_consensus = np.zeros((output_size, layer1))
        self.b2_consensus = np.zeros(output_size)

        # dual
        self.dual_W1 = np.zeros((layer1, input_size))
        self.dual_b1 = np.zeros(layer1)
        self.dual_W2 = np.zeros((output_size, layer1))
        self.dual_b2 = np.zeros(output_size)
        # ------------------------------------ COLLOCATION CONSTRAINTS ---------------------------------------

        self.model1.ode = ConstraintList()
        self.model2.ode = ConstraintList()

        for i in range(len(self.t1)):

            if i == 0:
                continue

            dy_dt_1 = sum(self.D1[i, j] * self.model1.y[j] for j in range(N))
            dy_dt_2 = sum(self.D2[i, j] * self.model2.y[j] for j in range(N))

            nn_input_1 = [self.model1.y[i]]
            nn_input_2 = [self.model2.y[i]]

            # add time and extra inputs
            if not self.time_invariant:
                nn_input_1.append(self.t1[i])
                nn_input_2.append(self.t2[i])

            if self.extra_input1 is not None:
                for input in self.extra_input1.T:
                    nn_input_1.append(input[i])
            if self.extra_input2 is not None:
                for input in self.extra_input2.T:
                    nn_input_2.append(input[i])

            nn_y_1 = self.nn_output(nn_input_1, self.model1)
            nn_y_2 = self.nn_output(nn_input_2, self.model2)
            #print("nn_y_1 ", nn_y_1)
            #print("dy_dt_1 ", dy_dt_1)
            # when are nn_y_1 and dy_dt_1 recalculated?
            self.model1.ode.add(nn_y_1 == dy_dt_1)
            self.model2.ode.add(nn_y_2 == dy_dt_2)

        self.update_objective()
    
    def update_objective(self):
            if hasattr(self.model1, 'obj'):
                self.model1.del_component('obj')
            if hasattr(self.model2, 'obj'):
                self.model2.del_component('obj')
        
            def _objective1(m):
                # data fit term
                data_fit = sum((m.y[i] - self.y_observed1[i])**2 for i in range(len(self.t1)))
                reg_smooth = sum((m.y[i] - m.y[i + 1])**2 for i in range(len(self.t1) - 1))

                # regularization term for weights and biases
                reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) +
                    sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) +
                    sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) +
                    sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

                if self.iter >= 1:
                    # ADMM penalty term for consensus
                    admm_penalty = (self.rho / 2) * (
                        sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j] / self.rho)**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                        sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i] / self.rho)**2 for i in range(self.layer_sizes[1])) +
                        sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j] / self.rho)**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                        sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i] / self.rho)**2 for i in range(self.layer_sizes[2]))
                    )
                else:
                    admm_penalty = 0

                return data_fit + self.penalty_lambda_reg * reg + self.penalty_lambda_smooth * reg_smooth + admm_penalty

            def _objective2(m):
                data_fit = sum((m.y[i] - self.y_observed2[i])**2 for i in range(len(self.t2)))
                reg_smooth = sum((m.y[i] - m.y[i + 1])**2 for i in range(len(self.t2) - 1))

                # regularization for weights and biases
                reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) +
                    sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) +
                    sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) +
                    sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

                if self.iter >= 1:
                    # ADMM penalty term for consensus
                    admm_penalty = (self.rho / 2) * (
                        sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j] / self.rho)**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                        sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i] / self.rho)**2 for i in range(self.layer_sizes[1])) +
                        sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j] / self.rho)**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                        sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i] / self.rho)**2 for i in range(self.layer_sizes[2]))
                    )
                else:
                    admm_penalty = 0
                    
                return data_fit + self.penalty_lambda_reg * reg + self.penalty_lambda_smooth * reg_smooth + admm_penalty

            self.model1.obj = Objective(rule=_objective1, sense=pyo.minimize)
            self.model2.obj = Objective(rule=_objective2, sense=pyo.minimize)
                        
    def nn_output(self, nn_input, m):
        # sum(a[i] * b[i] for i in range(len(a)))
        # hidden = sum()
        # print('inside nn_output')
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
        solver_info = {}

        # Solve model1
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result1 = solver.solve(self.model1, tee=True)

        # Extract solver information for model1
        solver_info['model1'] = {
            'solver_time': result1.solver.time,
            'termination_condition': result1.solver.termination_condition,
            'message': result1.solver.message
        }
        
        self.update_consensus_variables()
        self.update_dual_variables()

        # Solve model2
        result2 = solver.solve(self.model2, tee=True)

        # Extract solver information for model2
        solver_info['model2'] = {
            'solver_time': result2.solver.time,
            'termination_condition': result2.solver.termination_condition,
            'message': result2.solver.message
        }
        
        print(solver_info)
        
        self.update_consensus_variables()
        self.update_dual_variables()
        
        return solver_info
    
    def admm_solve(self, iterations=50, tol_primal=1e-3, tol_dual=1e-3):
        for i in range(iterations):
            print(''.join(['-' for i in range(100)]))
            print(f"ADMM Iteration {i+1}/{iterations}; {self.iter}")
            print(''.join(['-' for i in range(100)]))
            
            # Update model1 while keeping model2 fixed
            self.solve_model()
            self.update_objective()
            
            self.prev_W1_consensus = self.W1_consensus.copy()
            self.prev_b1_consensus = self.b1_consensus.copy()
            self.prev_W2_consensus = self.W2_consensus.copy()
            self.prev_b2_consensus = self.b2_consensus.copy()
            
            primal_residual = self.compute_primal_residual()
            # dual_residual = self.compute_dual_residual()
            
            print(f"Primal Residual: {primal_residual}")

            # check for convergence
            if primal_residual < tol_primal:
                print(''.join(['*' for i in range(100)]))
                print(f"Converged at iteration {i+1}")
                print(''.join(['*' for i in range(100)]))
                return
            
            self.iter += 1
        
        print(''.join(['*' for i in range(100)]))
        print(f"Model did not converge.")
        print(f"Primal Residual: {primal_residual}")
        print(''.join(['*' for i in range(100)]))
        
    def update_dual_variables(self):
        for i in range(self.layer_sizes[1]):
            for j in range(self.layer_sizes[0]):
                self.dual_W1[i, j] += 0.5 * (self.model1.W1[i, j]() - self.W1_consensus[i, j])
                self.dual_W1[i, j] += 0.5 * (self.model2.W1[i, j]() - self.W1_consensus[i, j])
        for i in range(self.layer_sizes[1]):
            self.dual_b1[i] += 0.5 * (self.model1.b1[i]() - self.b1_consensus[i])
            self.dual_b1[i] += 0.5 * (self.model2.b1[i]() - self.b1_consensus[i])
        for i in range(self.layer_sizes[2]):
            for j in range(self.layer_sizes[1]):
                self.dual_W2[i, j] += 0.5 * (self.model1.W2[i, j]() - self.W2_consensus[i, j])
                self.dual_W2[i, j] += 0.5 * (self.model2.W2[i, j]() - self.W2_consensus[i, j])
        for i in range(self.layer_sizes[2]):
            self.dual_b2[i] += 0.5 * (self.model1.b2[i]() - self.b2_consensus[i])
            self.dual_b2[i] += 0.5 * (self.model2.b2[i]() - self.b2_consensus[i])
    
    def update_consensus_variables(self):
        self.W1_consensus = (np.array([[self.model1.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])]) + 
                             np.array([[self.model2.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])])) / 2
        self.b1_consensus = (np.array([self.model1.b1[i].value for i in range(self.layer_sizes[1])]) +
                             np.array([self.model2.b1[i].value for i in range(self.layer_sizes[1])])) / 2
        self.W2_consensus = (np.array([[self.model1.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])]) + 
                             np.array([[self.model2.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])])) / 2
        self.b2_consensus = (np.array([self.model1.b2[i].value for i in range(self.layer_sizes[2])]) + 
                             np.array([self.model2.b2[i].value for i in range(self.layer_sizes[2])])) / 2
    
    def to_array(self, pyomo_var, shape):
        return np.array([[pyomo_var[i, j].value for j in range(shape[1])] for i in range(shape[0])])

    def to_vector(self, pyomo_var, size):
        return np.array([pyomo_var[i].value for i in range(size)])
    
    def get_model_vars(self, model):
        return {
            'W1': self.to_array(model.W1, (self.layer_sizes[1], self.layer_sizes[0])),
            'b1': self.to_vector(model.b1, self.layer_sizes[1]),
            'W2': self.to_array(model.W2, (self.layer_sizes[2], self.layer_sizes[1])),
            'b2': self.to_vector(model.b2, self.layer_sizes[2])
        }

    def compute_primal_residual(self):
        model1_vars = self.get_model_vars(self.model1)
        model2_vars = self.get_model_vars(self.model2)

        primal_residuals = [
            np.linalg.norm(model1_vars[var] - getattr(self, f"{var}_consensus")) + np.linalg.norm(model2_vars[var] - getattr(self, f"{var}_consensus"))
            for var in ['W1', 'b1', 'W2', 'b2']
        ]

        return sum(primal_residuals) 
    
    def extract_solution(self):
        y1 = np.array([value(self.model1.y[i]) for i in self.model1.t])
        y2 = np.array([value(self.model2.y[i]) for i in self.model2.t])
        return np.concatenate([y1, y2])

    def extract_derivative(self):
        dy_dt_1, dy_dt_2 = [], []
        for i in range(len(self.t1)):
            dy_dt_1_i = sum(self.D1[i, j] * pyo.value(self.model1.y[j]) for j in range(len(self.t1)))
            dy_dt_1.append(dy_dt_1_i)
        
        for i in range(len(self.t2)):
            dy_dt_2_i = sum(self.D2[i, j] * pyo.value(self.model2.y[j]) for j in range(len(self.t2)))
            dy_dt_2.append(dy_dt_2_i)
        
        return np.array(dy_dt_1), np.array(dy_dt_2)

    def extract_weights(self, m = None):
        weights = {}
        if m is None:
            weights['W1'], weights['b1'] = self.W1_consensus, self.b1_consensus
            weights['W2'], weights['b2'] = self.W2_consensus, self.b2_consensus
        else:
            W1 = np.array([[value(m.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
            b1 = np.array([value(m.b1[j]) for j in range(self.layer_sizes[1])])
            W2 = np.array([[value(m.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
            b2 = np.array([value(m.b2[j]) for j in range(self.layer_sizes[2])])   
            weights['W1'], weights['b1'], weights['W2'], weights['b2'] = W1, b1, W2, b2

        return weights
    
    def predict(self, input, weights):
        # Extract weights from both models
        if weights == 'both':
            weights1 = self.extract_weights(self.model1)
            weights2 = self.extract_weights(self.model2)

            # Average the weights and biases
            W1 = (weights1['W1'] + weights2['W1']) / 2
            b1 = (weights1['b1'] + weights2['b1']) / 2
            W2 = (weights1['W2'] + weights2['W2']) / 2
            b2 = (weights1['b2'] + weights2['b2']) / 2
        elif weights == 'consensus':
            W1, b1, W2, b2 = self.W1_consensus, self.b1_consensus, self.W2_consensus, self.b2_consensus
        else:
            if weights == 'model1':
                weights1 = self.extract_weights(self.model1)
                W1, b1, W2, b2 = weights1['W1'], weights1['b1'], weights1['W2'], weights1['b2']
            elif weights == 'model2':
                weights2 = self.extract_weights(self.model2)
                W1, b1, W2, b2 = weights2['W1'], weights2['b1'], weights2['W2'], weights2['b2']
            else:
                raise ValueError("Not a valid weights argument.")

        # Compute the hidden layer output
        hidden = jnp.tanh(jnp.dot(W1, input) + b1)

        # Compute the output layer
        outputs = jnp.dot(W2, hidden) + b2

        return outputs

    def neural_ode(self, y0, t, extra_args = None, weights = 'both'):
        """
        Args:
            weights (str, optional): 'both', 'model1', 'model2', 'consensus'. Defaults to 'both'.
        """
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)

            if args is not None:
                extra_inputs, t_all = args
                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    if extra_inputs.ndim == 2:
                        interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(t, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, extra_inputs)

            result = self.predict(input, weights)
            return result

        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-8)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            term,  # function
            solver,
            t0=t[0],
            t1=t[-1],
            # dt0=1e-3,
            dt0 = t[-1] - t[0],
            y0=y0,
            args=extra_args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
            #adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=100),
            #max_steps = 10000
        )
        # print("solution.ts", solution.ts)
        return solution.ys
