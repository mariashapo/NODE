import numpy as np
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Constraint, ConstraintList, Objective, SolverFactory
from pyomo.dae import ContinuousSet, DerivativeVar

"""
This method is a Pyomo implementation of the ADMM algorithm for training a neural network.
The approach is based on having 2 models, one for each data batch.
The penalty term is added to both objective functions to ensure the parameters are similar.
"""

class NeuralODEPyomoADMM:
    def __init__(self, y_observed, t, layer_sizes, extra_input = None, penalty_lambda_reg=0.1, penalty_rho=1.0):
        # split the data into 2 batches
        midpoint = len(t) // 2
        self.t1 = t[:midpoint]
        self.t2 = t[midpoint:]
        self.y_observed1 = y_observed[:midpoint]
        self.y_observed2 = y_observed[midpoint:]
        
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
        self.model1.y1 = Var(self.model1.t, domain=pyo.Reals, 
                    initialize=lambda m, t: self.y_observed1[(np.abs(self.t1 - t)).argmin()], 
                    bounds=(lower_bound, upper_bound))
        
        self.model2.y2 = Var(self.model2.t, domain=pyo.Reals, 
                    initialize=lambda m, t: self.y_observed2[(np.abs(self.t2 - t)).argmin()], 
                    bounds=(lower_bound, upper_bound))
        
        self.model1.dy1_dt = DerivativeVar(self.model1.y1, wrt=self.model1.t1)
        self.model2.dy2_dt = DerivativeVar(self.model2.y2, wrt=self.model2.t2)
        
        input_size, layer1, output_size = self.layer_sizes[0], self.layer_sizes[1], self.layer_sizes[2]
        
        # initialize weights and biases for both models
        weight_bounds = (-100.0, 100.0)
        
        # model 1
        self.model1.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model1.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model1.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model1.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
        
        # model 2
        self.model1.W1 = Var(range(layer1), range(input_size), initialize=lambda m, i, j: self.initialize_weights((layer1, input_size))[i, j], bounds=weight_bounds)
        self.model1.b1 = Var(range(layer1), initialize=lambda m, i: self.initialize_biases(layer1)[i], bounds=weight_bounds)
        self.model1.W2 = Var(range(output_size), range(layer1), initialize=lambda m, i, j: self.initialize_weights((output_size, layer1))[i, j], bounds=weight_bounds)
        self.model1.b2 = Var(range(output_size), initialize=lambda m, i: self.initialize_biases(output_size)[i], bounds=weight_bounds)
        
        
    def setup_constraints(self):
        def _con1_y1(m, t_i):
            if t_i == self.t[0]:
                return Constraint.Skip  
            
            # define the correct neural net input
            nn_input = [m.y1[t_i]]  
            if not self.time_invariant:
                nn_input.append(t_i)
        
            if self.extra_input1 is not None:
                index = (np.abs(self.t1 - t_i)).argmin()
                for input in self.extra_input1.T:
                    nn_input.append(input[index])

            nn_y1 = self.nn_output(nn_input, m)
            return nn_y1 == m.dy1_dt[t_i] 

        def _con1_y2(m, t_i):
            if t_i == self.t[0]:
                return Constraint.Skip 
            
            nn_input = [m.y2[t_i]]  
            if not self.time_invariant:
                nn_input.append(t_i)
        
            if self.extra_input2 is not None:
                index = (np.abs(self.t2 - t_i)).argmin()
                for input in self.extra_input2.T:
                    nn_input.append(input[index])

            nn_y2 = self.nn_output(nn_input, m)
            return nn_y2 == model.dy2_dt[t_i] 
        
        # add constraints for both models
        self.model1.con1_y1 = Constraint(self.model1.t1, rule=_con1_y1)
        self.model2.con1_y2 = Constraint(self.model2.t2, rule=_con1_y2)

    def nn_output(self, nn_input, m):
        # this function remains shared since the model object is passed in
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

    def _objective1(self, m):
        # access model y parameter at the correct time point
        # and y_observed at the correct index;
        data_fit =  sum((m.y1[t_i] - self.y_observed1[i])**2 for i, t_i in enumerate(self.t1))
        # next index (i+1) is used to access the next time point
        reg_smooth = sum((m.y1[self.t1[i+1]] - m.y[self.t1[i]])**2 for i in range(len(self.t1) - 1))
        return data_fit + self.penalty_lambda_reg * reg_smooth
    
    def _objective2(self, m):
        data_fit =  sum((m.y2[t_i] - self.y_observed2[i])**2 for i, t_i in enumerate(self.t2))
        reg_smooth = sum((m.y2[self.t1[i+1]] - m.y[self.t2[i]])**2 for i in range(len(self.t2) - 1))
        return data_fit + self.penalty_lambda_reg * reg_smooth
    
    def admm_solve(self, iterations=50):
        solver = SolverFactory('ipopt')
        tolerance = 1e-4  # Convergence tolerance
        converged = False
        
        

        for i in range(iterations):
            # batch 1
            self.model.obj = Objective(expr=self._objective1(self.model), sense=pyo.minimize)
            solver.solve(self.model)
            # remove the objective to redefine in the next step
            self.model.del_component(self.model.obj)  

            # batch 2
            self.model.obj = Objective(expr=self._objective2(self.model), sense=pyo.minimize)
            solver.solve(self.model)
            self.model.del_component(self.model.obj)
            
            self.update_dual_variables()

            # Check for convergence (this can be defined more elaborately depending on the problem specifics)
            if self.check_convergence(tolerance):
                converged = True
                break

        return converged

    def update_dual_variables(self):
        # Update the dual variables using the discrepancy between y1 and y2
        for t in self.t1.union(self.t2):
            if t in self.t1 and t in self.t2:  # Ensure t is in both time sets
                discrepancy = self.model.y1[t].value - self.model.y2[t].value
                self.model.lambda_[t].set_value(self.model.lambda_[t].value + self.penalty_rho * discrepancy)

    def check_convergence(self, tolerance):
        # Check if the maximum absolute dual update is below the tolerance
        max_update = max(abs(self.model.lambda_[t].value - (self.model.y1[t].value - self.model.y2[t].value) * self.penalty_rho)
                        for t in self.t1.union(self.t2) if t in self.t1 and t in self.t2)
        return max_update < tolerance


# Example usage:
y_observed = np.random.rand(100)
t = np.linspace(0, 10, 100)
first_derivative_matrix = np.random.rand(100, 100)  # Example matrix
layer_sizes = {'theta': 3}

model = NeuralODEPyomoADMM(y_observed, t, first_derivative_matrix, layer_sizes)
model.admm_solve()
