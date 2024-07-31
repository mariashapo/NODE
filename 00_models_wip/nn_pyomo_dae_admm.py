import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory
from pyomo.dae import ContinuousSet, DerivativeVar

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax as dfx

class NeuralODEPyomoADMM:
    def __init__(self, y_observed, t, layer_sizes, extra_input=None, penalty_lambda_reg=0.1, penalty_rho=1.0, params = None):
        # split the data into 2 batches
        midpoint = len(t) // 2
        self.t1 = t[:midpoint]
        self.t2 = t[midpoint:]
        self.y_observed1 = y_observed[:midpoint]
        self.y_observed2 = y_observed[midpoint:]
        self.w_init_method = 'xavier'
        self.act_func = 'tanh'
        self.discretization_scheme = 'LAGRANGE-RADAU'
        self.time_invariant = True
        self.ncp = 3
        self.params = params
        self.iter = 0

        if extra_input is not None:
            self.extra_input1 = extra_input[:midpoint]
            self.extra_input2 = extra_input[midpoint:]
        else:
            self.extra_input1 = None
            self.extra_input2 = None

        self.layer_sizes = layer_sizes
        self.penalty_lambda_reg = penalty_lambda_reg
        self.penalty_rho = penalty_rho

        self.model1 = ConcreteModel()
        self.model2 = ConcreteModel()

        self.setup_variables()
        self.setup_constraints()

        # Initialize dual variables
        # self.dual_vars_W1 = np.zeros((layer_sizes[1], layer_sizes[0]))
        # self.dual_vars_b1 = np.zeros(layer_sizes[1])
        # self.dual_vars_W2 = np.zeros((layer_sizes[2], layer_sizes[1]))
        # self.dual_vars_b2 = np.zeros(layer_sizes[2])

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

    def setup_variables(self):
        self.model1.t1 = ContinuousSet(initialize=self.t1)
        self.model2.t2 = ContinuousSet(initialize=self.t2)

        lower_bound = -5.0
        upper_bound = 5.0

        # initialize y1 and y2 with the observed data
        self.model1.y1 = Var(self.model1.t1, domain=pyo.Reals,
                             initialize=lambda m, t: self.y_observed1[(np.abs(self.t1 - t)).argmin()],
                             bounds=(lower_bound, upper_bound))

        self.model2.y2 = Var(self.model2.t2, domain=pyo.Reals,
                             initialize=lambda m, t: self.y_observed2[(np.abs(self.t2 - t)).argmin()],
                             bounds=(lower_bound, upper_bound))

        input_size, layer1, output_size = self.layer_sizes[0], self.layer_sizes[1], self.layer_sizes[2]

        weight_bounds = (-100.0, 100.0)

        # model 1
        self.model1.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model1.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model1.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model1.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

        # model 2
        self.model2.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model2.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model2.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model2.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)

    def setup_constraints(self):
        
        self.model1.dy1_dt = DerivativeVar(self.model1.y1, wrt=self.model1.t1)
        self.model2.dy2_dt = DerivativeVar(self.model2.y2, wrt=self.model2.t2)
        
        def _con1_y1(m, t_i):
            if t_i == self.t1[0]:
                return Constraint.Skip

            nn_input = [m.y1[t_i]]
            if self.extra_input1 is not None:
                index = (np.abs(self.t1 - t_i)).argmin()
                for input in self.extra_input1.T:
                    nn_input.append(input[index])

            nn_y1 = self.nn_output(nn_input, m)
            return nn_y1 == m.dy1_dt[t_i]

        def _con2_y1(m, t_i):
            if t_i == self.t2[0]:
                return Constraint.Skip

            nn_input = [m.y2[t_i]]
            if self.extra_input2 is not None:
                index = (np.abs(self.t2 - t_i)).argmin()
                for input in self.extra_input2.T:
                    nn_input.append(input[index])

            nn_y2 = self.nn_output(nn_input, m)
            return nn_y2 == m.dy2_dt[t_i]

        # add constraints for both models
        self.model1.con1_y1 = Constraint(self.model1.t1, rule=_con1_y1)
        #self.model2.con2_y1 = Constraint(self.model2.t2, rule=_con2_y1)

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
        
        return outputs

    def update_dual_variables(self):
        # W1
        for i in range(self.layer_sizes[1]):
            for j in range(self.layer_sizes[0]):
                self.dual_vars_W1[i, j] += self.model1.W1[i, j]() - self.model2.W1[i, j]()
        # b1
        for i in range(self.layer_sizes[1]):
            self.dual_vars_b1[i] += self.model1.b1[i]() - self.model2.b1[i]() 
        # W2
        for i in range(self.layer_sizes[2]):
            for j in range(self.layer_sizes[1]):
                self.dual_vars_W2[i, j] += self.model1.W2[i, j]() - self.model2.W2[i, j]()
        # b2
        for i in range(self.layer_sizes[2]):
            self.dual_vars_b2[i] += self.model1.b2[i]() - self.model2.b2[i]()

    def check_convergence(self, tolerance):
        return True
        # Primal residuals for the weights and biases consistency
        primal_residuals_W1 = sum((self.model1.W1[i, j]() - self.model2.W1[i, j]())**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0]))
        primal_residuals_b1 = sum((self.model1.b1[i]() - self.model2.b1[i]())**2 for i in range(self.layer_sizes[1]))
        primal_residuals_W2 = sum((self.model1.W2[i, j]() - self.model2.W2[i, j]())**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1]))
        primal_residuals_b2 = sum((self.model1.b2[i]() - self.model2.b2[i]())**2 for i in range(self.layer_sizes[2]))

        primal_residual = np.sqrt(primal_residuals_W1 + primal_residuals_b1 + primal_residuals_W2 + primal_residuals_b2)

        return primal_residual < tolerance 

    def admm_solve(self, iterations=50, tolerance = 1e-4):
        # solver = SolverFactory('ipopt')
        converged = False

        def _objective1(m):
            data_fit = sum((m.y1[t_i] - self.y_observed1[i])**2 for i, t_i in enumerate(self.t1))
            reg_smooth = sum((m.y1[self.t1[i+1]] - m.y1[self.t1[i]])**2 for i in range(len(self.t1) - 1))
            
            if self.iter >= 2:
                print('ADMM entry')
                penalty = (self.penalty_rho / 2) * (
                    sum((m.W1[i, j] - self.model2.W1[i, j] + self.dual_vars_W1[i, j])**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                    sum((m.b1[i] - self.model2.b1[i] + self.dual_vars_b1[i])**2 for i in range(self.layer_sizes[1])) +
                    sum((m.W2[i, j] - self.model2.W2[i, j] + self.dual_vars_W2[i, j])**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                    sum((m.b2[i] - self.model2.b2[i] + self.dual_vars_b2[i])**2 for i in range(self.layer_sizes[2]))
                )
                return data_fit + self.penalty_lambda_reg * reg_smooth + penalty
            else:
                return data_fit + self.penalty_lambda_reg * reg_smooth

        def _objective2(m):
            data_fit = sum((m.y2[t_i] - self.y_observed2[i])**2 for i, t_i in enumerate(self.t2))
            reg_smooth = sum((m.y2[self.t2[i+1]] - m.y2[self.t2[i]])**2 for i in range(len(self.t2) - 1))
            if self.iter >= 2:
                penalty = (self.penalty_rho / 2) * (
                    sum((m.W1[i, j] - self.model1.W1[i, j] + self.dual_vars_W1[i, j])**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                    sum((m.b1[i] - self.model1.b1[i] + self.dual_vars_b1[i])**2 for i in range(self.layer_sizes[1])) +
                    sum((m.W2[i, j] - self.model1.W2[i, j] + self.dual_vars_W2[i, j])**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                    sum((m.b2[i] - self.model1.b2[i] + self.dual_vars_b2[i])**2 for i in range(self.layer_sizes[2]))
                )
                return data_fit + self.penalty_lambda_reg * reg_smooth + penalty
            else:
                return data_fit + self.penalty_lambda_reg * reg_smooth
            
        for i in range(iterations):
            # batch 1
            self.model1.obj = Objective(expr=_objective1, sense=pyo.minimize)
            self.solve_model(self.model1, self.model1.t1)

            # batch 2
            # self.model2.obj = Objective(expr=_objective2, sense=pyo.minimize)
            # self.solve_model(self.model2, self.model2.t2)

            if self.check_convergence(tolerance):
                converged = True
                print(f"Converged at iteration {i}")
                break

        return converged
    
    def solve_model(self, model, t):
        #if self.iter >= 0:
        if True:
            if self.discretization_scheme == 'LAGRANGE-RADAU':
                discretizer = pyo.TransformationFactory('dae.collocation')
                discretizer.apply_to(model, nfe=len(t)-1, ncp=self.ncp, scheme='LAGRANGE-RADAU')
                
        # self.iter+=1 
        # solve the model
        solver = SolverFactory('ipopt')
        
        if self.params is not None:
            for key, value in self.params.items():
                solver.options[key] = value
        result = solver.solve(model, tee=True)

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
    
    def extract_direct_solutions(self):
        solutions = {
            'model1': {
                'W1': np.array([[self.model1.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])]),
                'b1': np.array([self.model1.b1[i].value for i in range(self.layer_sizes[1])]),
                'W2': np.array([[self.model1.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])]),
                'b2': np.array([self.model1.b2[i].value for i in range(self.layer_sizes[2])]),
                'y1': np.array([self.model1.y1[t_i].value for t_i in self.t1])
            },
            'model2': {
                'W1': np.array([[self.model2.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])]),
                'b1': np.array([self.model2.b1[i].value for i in range(self.layer_sizes[1])]),
                'W2': np.array([[self.model2.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])]),
                'b2': np.array([self.model2.b2[i].value for i in range(self.layer_sizes[2])]),
                'y2': np.array([self.model2.y2[t_i].value for t_i in self.t2])
            }
        }
        return solutions
    
    def extract_derivative(self):
        dy1_dt, dy2_dt = [], []
        for t in self.model1.t1:
            try:
                dy1_dt.append((t, pyo.value(self.model1.dy1_dt[t])))
            except ValueError:
                dy1_dt.append((t, None)) 
                
        """for t in self.model2.t2:
            try:
                dy2_dt.append((t, pyo.value(self.model2.dy2_dt[t])))
            except ValueError:
                dy2_dt.append((t, None)) """
                
        dy1_dt = np.array(dy1_dt)
        # dy2_dt = np.array(dy2_dt)
        
        return [dy1_dt, dy2_dt]

    def set_nn_parameters(self, solutions):
        averaged = False
        if averaged:
            self.W1 = (solutions['model1']['W1'] + solutions['model2']['W1']) / 2
            self.b1 = (solutions['model1']['b1'] + solutions['model2']['b1']) / 2
            self.W2 = (solutions['model1']['W2'] + solutions['model2']['W2']) / 2
            self.b2 = (solutions['model1']['b2'] + solutions['model2']['b2']) / 2
        else:
            self.W1 = solutions['model1']['W1']
            self.b1 = solutions['model1']['b1']
            self.W2 = solutions['model1']['W2']
            self.b2 = solutions['model1']['b2']
    
    
    def predict(self, input):
        solution = self.extract_direct_solutions()
        self.set_nn_parameters(solution)
        
        input_size, hidden_size, output_size = self.layer_sizes
        
        # compute the hidden layer output
        hidden = [sum(self.W1[i, j] * input[j] for j in range(input_size)) + self.b1[i] for i in range(hidden_size)]
        
        # activation function
        if self.act_func == "tanh":
            hidden = [jnp.tanh(h) for h in hidden]
        elif self.act_func == "sigmoid":
            hidden = [1 / (1 + jnp.exp(-h)) for h in hidden]
        elif self.act_func == "softplus":
            hidden = [jnp.log(1 + jnp.exp(h)) for h in hidden]
        
        # compute the output layer
        outputs = [sum(self.W2[i, j] * hidden[j] for j in range(hidden_size)) + self.b2[i] for i in range(output_size)]
        
        return outputs[0] 

    def neural_ode(self, y0, t, extra_args=None): 
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)

            if args is not None:
                extra_inputs, t_all = args
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
        
        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            term,  # function
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=1e-3,
            y0=y0,
            args=extra_args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
        )

        return solution.ys
    
    
"""
L, R = 0, 400

ys = np.atleast_2d(y_noisy_vdp[L:R, 0]).T
ts = np.array(t_vdp)[L:R]
Xs = np.atleast_2d(y_noisy_vdp[L:R, 1]).T

import nn_pyomo_admm_2b
importlib.reload(nn_pyomo_admm_2b)
NeuralODEPyomoADMM = nn_pyomo_admm_2b.NeuralODEPyomoADMM

tol = 1e-7
params = {"tol":tol, 
        "dual_inf_tol": tol, "compl_inf_tol": tol, "constr_viol_tol": tol, 
        # "acceptable_tol": 1e-15, "acceptable_constr_viol_tol": 1e-15, "acceptable_dual_inf_tol": 1e-15, "acceptable_compl_inf_tol": 1e-15, "acceptable_iter": 0, 
        "halt_on_ampl_error" : 'yes', "print_level": 5, "max_iter": 500}

layer_widths = [2, 20, 1]

neural_ode_admm = NeuralODEPyomoADMM(ys, ts, layer_widths, extra_input=Xs, penalty_lambda_reg = 10, penalty_rho=0.01, params = params)
converged = neural_ode_admm.admm_solve(iterations=50, tolerance = 1e-1)

if converged:
    print("ADMM solver converged.")
else:
    print("ADMM solver did not converge.")
    
solution = neural_ode_admm.extract_direct_solutions()
y1, y2 = solution['model1']['y1'], solution['model2']['y2']
#y_full = np.concatenate([y1, y2])
    
"""
