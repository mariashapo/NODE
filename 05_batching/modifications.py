import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, SolverFactory, value, RangeSet
import jax
import jax.numpy as jnp
import diffrax as dfx

np.random.seed(42)

class NeuralODEPyomoADMM:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, rho=1.0,
                 penalty_lambda_reg=0.01, penalty_lambda_smooth=0.0, act_func="tanh", w_init_method="random", params=None, y_init=None):
        
        self.initialize_parameters(y_observed, t, first_derivative_matrix, layer_sizes, time_invariant, extra_input, rho,
                                   penalty_lambda_reg, penalty_lambda_smooth, act_func, w_init_method, params, y_init)
        self.model1, self.model2 = ConcreteModel(), ConcreteModel()
        self.create_submodels()



    def update_objective(self):
        if hasattr(self.model1, 'obj'):
            self.model1.del_component('obj')
        if hasattr(self.model2, 'obj'):
            self.model2.del_component('obj')
        self.model1.obj = Objective(rule=self.create_objective(self.model1, self.y_observed1, self.t1), sense=pyo.minimize)
        self.model2.obj = Objective(rule=self.create_objective(self.model2, self.y_observed2, self.t2), sense=pyo.minimize)

    def create_objective(self, model, y_observed, t):
        def _objective(m):
            data_fit = sum((m.y[i] - y_observed[i])**2 for i in range(len(t)))
            reg_smooth = sum((m.y[i] - m.y[i + 1])**2 for i in range(len(t) - 1))
            reg = self.compute_regularization_term(m)
            admm_penalty = self.compute_admm_penalty(m) if self.iter >= 1 else 0
            return data_fit + self.penalty_lambda_reg * reg + self.penalty_lambda_smooth * reg_smooth + admm_penalty
        return _objective

    def compute_regularization_term(self, m):
        layer1, input_size, output_size = self.layer_sizes[1], self.layer_sizes[0], self.layer_sizes[2]
        reg = (sum(m.W1[j, k]**2 for j in range(layer1) for k in range(input_size)) +
               sum(m.W2[j, k]**2 for j in range(output_size) for k in range(layer1)) +
               sum(m.b1[j]**2 for j in range(layer1)) +
               sum(m.b2[j]**2 for j in range(output_size)))
        return reg

    def compute_admm_penalty(self, m):
        layer1, input_size, output_size = self.layer_sizes[1], self.layer_sizes[0], self.layer_sizes[2]
        admm_penalty = (self.rho / 2) * (
            sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j] / self.rho)**2 for i in range(layer1) for j in range(input_size)) +
            sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i] / self.rho)**2 for i in range(layer1)) +
            sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j] / self.rho)**2 for i in range(output_size) for j in range(layer1)) +
            sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i] / self.rho)**2 for i in range(output_size))
        )
        return admm_penalty

    def nn_output(self, nn_input, m):
        hidden = self.apply_activation_function(jnp.dot(m.W1, nn_input) + m.b1)
        outputs = jnp.dot(m.W2, hidden) + m.b2
        return outputs

    def apply_activation_function(self, hidden):
        epsilon = 1e-10
        if self.act_func == "tanh":
            return jnp.tanh(hidden)
        if self.act_func == "sigmoid":
            return 1 / (1 + jnp.exp(-hidden) + epsilon)
        if self.act_func == "softplus":
            return jnp.log(1 + jnp.exp(hidden) + epsilon)
        raise ValueError("Unsupported activation function. Use 'tanh', 'sigmoid', or 'softplus'.")

    def solve_model(self):
        solver = SolverFactory('ipopt')
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result1 = solver.solve(self.model1, tee=True)
        solver_info = self.extract_solver_info(result1, 'model1')
        self.update_consensus_variables()
        self.update_dual_variables()
        result2 = solver.solve(self.model2, tee=True)
        solver_info.update(self.extract_solver_info(result2, 'model2'))
        print(solver_info)
        self.update_consensus_variables()
        self.update_dual_variables()
        return solver_info

    def extract_solver_info(self, result, model_name):
        return {model_name: {
            'solver_time': result.solver.time,
            'termination_condition': result.solver.termination_condition,
            'message': result.solver.message
        }}

    def admm_solve(self, iterations=50, tol_primal=1e-3, tol_dual=1e-3):
        for i in range(iterations):
            print('-' * 100)
            print(f"ADMM Iteration {i + 1}/{iterations}; {self.iter}")
            print('-' * 100)
            self.solve_model()
            self.update_objective()
            primal_residual = self.compute_primal_residual()
            print(f"Primal Residual: {primal_residual}")
            if primal_residual < tol_primal:
                print('*' * 100)
                print(f"Converged at iteration {i + 1}")
                print('*' * 100)
                return
            self.iter += 1
        print('*' * 100)
        print(f"Model did not converge. Primal Residual: {primal_residual}")
        print('*' * 100)

    def update_dual_variables(self):
        self.dual_W1 += 0.5 * (self.to_array(self.model1.W1) - self.W1_consensus + self.to_array(self.model2.W1) - self.W1_consensus)
        self.dual_b1 += 0.5 * (self.to_vector(self.model1.b1) - self.b1_consensus + self.to_vector(self.model2.b1) - self.b1_consensus)
        self.dual_W2 += 0.5 * (self.to_array(self.model1.W2) - self.W2_consensus + self.to_array(self.model2.W2) - self.W2_consensus)
        self.dual_b2 += 0.5 * (self.to_vector(self.model1.b2) - self.b2_consensus + self.to_vector(self.model2.b2) - self.b2_consensus)

    def update_consensus_variables(self):
        self.W1_consensus = (self.to_array(self.model1.W1) + self.to_array(self.model2.W1)) / 2
        self.b1_consensus = (self.to_vector(self.model1.b1) + self.to_vector(self.model2.b1)) / 2
        self.W2_consensus = (self.to_array(self.model1.W2) + self.to_array(self.model2.W2)) / 2
        self.b2_consensus = (self.to_vector(self.model1.b2) + self.to_vector(self.model2.b2)) / 2

    def to_array(self, pyomo_var):
        return np.array([[pyomo_var[i, j].value for j in range(pyomo_var.index_set().dimen)] for i in range(pyomo_var.index_set().dimen)])

    def to_vector(self, pyomo_var):
        return np.array([pyomo_var[i].value for i in range(pyomo_var.index_set().dimen)])

    def get_model_vars(self, model):
        return {
            'W1': self.to_array(model.W1),
            'b1': self.to_vector(model.b1),
            'W2': self.to_array(model.W2),
            'b2': self.to_vector(model.b2)
        }

    def compute_primal_residual(self):
        model1_vars = self.get_model_vars(self.model1)
        model2_vars = self.get_model_vars(self.model2)
        primal_residuals = [np.linalg.norm(model1_vars[var] - getattr(self, f"{var}_consensus")) + np.linalg.norm(model2_vars[var] - getattr(self, f"{var}_consensus"))
                            for var in ['W1', 'b1', 'W2', 'b2']]
        return sum(primal_residuals)

    def extract_solution(self):
        y1 = np.array([value(self.model1.y[i]) for i in self.model1.t])
        y2 = np.array([value(self.model2.y[i]) for i in self.model2.t])
        return np.concatenate([y1, y2])

    def extract_derivative(self):
        dy_dt_1 = [sum(self.D1[i, j] * pyo.value(self.model1.y[j]) for j in range(len(self.t1))) for i in range(len(self.t1))]
        dy_dt_2 = [sum(self.D2[i, j] * pyo.value(self.model2.y[j]) for j in range(len(self.t2))) for i in range(len(self.t2))]
        return np.array(dy_dt_1), np.array(dy_dt_2)

    def extract_weights(self, m=None):
        if m is None:
            return {'W1': self.W1_consensus, 'b1': self.b1_consensus, 'W2': self.W2_consensus, 'b2': self.b2_consensus}
        W1 = np.array([[value(m.W1[j, k]) for k in range(self.layer_sizes[0])] for j in range(self.layer_sizes[1])])
        b1 = np.array([value(m.b1[j]) for j in range(self.layer_sizes[1])])
        W2 = np.array([[value(m.W2[j, k]) for k in range(self.layer_sizes[1])] for j in range(self.layer_sizes[2])])
        b2 = np.array([value(m.b2[j]) for j in range(self.layer_sizes[2])])
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def predict(self, input, weights):
        if weights == 'both':
            weights1, weights2 = self.extract_weights(self.model1), self.extract_weights(self.model2)
            W1, b1, W2, b2 = (weights1['W1'] + weights2['W1']) / 2, (weights1['b1'] + weights2['b1']) / 2, (weights1['W2'] + weights2['W2']) / 2, (weights1['b2'] + weights2['b2']) / 2
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
        hidden = jnp.tanh(jnp.dot(W1, input) + b1)
        outputs = jnp.dot(W2, hidden) + b2
        return outputs

    def neural_ode(self, y0, t, extra_args=None, weights='both'):
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)
            if args is not None:
                extra_inputs, t_all = args
                input = self.add_interpolated_inputs(input, extra_inputs, t, t_all)
            return self.predict(input, weights)
        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-8, atol=1e-8)
        saveat = dfx.SaveAt(ts=t)
        solution = dfx.diffeqsolve(term, solver, t0=t[0], t1=t[-1], dt0=t[-1] - t[0], y0=y0, args=extra_args, stepsize_controller=stepsize_controller, saveat=saveat)
        return solution.ys

    def add_interpolated_inputs(self, input, extra_inputs, t, t_all):
        if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
            if extra_inputs.ndim == 2:
                interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                input = jnp.append(input, interpolated_inputs)
            elif extra_inputs.ndim == 1:
                interpolated_input = jnp.interp(t, t_all, extra_inputs)
                input = jnp.append(input, interpolated_input)
        else:
            input = jnp.append(input, extra_inputs)
        return input
