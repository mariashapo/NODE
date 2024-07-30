
def update_dual_variables(self):
        for i in range(self.layer_sizes[1]):
            for j in range(self.layer_sizes[0]):
                self.dual_W1[i, j] += self.model1.W1[i, j]() - self.W1_consensus[i, j]
                self.dual_W1[i, j] += self.model2.W1[i, j]() - self.W1_consensus[i, j]
        for i in range(self.layer_sizes[1]):
            self.dual_b1[i] += self.model1.b1[i]() - self.b1_consensus[i]
            self.dual_b1[i] += self.model2.b1[i]() - self.b1_consensus[i]
        for i in range(self.layer_sizes[2]):
            for j in range(self.layer_sizes[1]):
                self.dual_W2[i, j] += self.model1.W2[i, j]() - self.W2_consensus[i, j]
                self.dual_W2[i, j] += self.model2.W2[i, j]() - self.W2_consensus[i, j]
        for i in range(self.layer_sizes[2]):
            self.dual_b2[i] += self.model1.b2[i]() - self.b2_consensus[i]
            self.dual_b2[i] += self.model2.b2[i]() - self.b2_consensus[i]


def update_consensus_variables(self):
    self.W1_consensus = (np.array([[self.model1.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])]) + 
                            np.array([[self.model2.W1[i, j].value for j in range(self.layer_sizes[0])] for i in range(self.layer_sizes[1])])) / 2
    self.b1_consensus = (np.array([self.model1.b1[i].value for i in range(self.layer_sizes[1])]) +
                            np.array([self.model2.b1[i].value for i in range(self.layer_sizes[1])])) / 2
    self.W2_consensus = (np.array([[self.model1.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])]) + 
                            np.array([[self.model2.W2[i, j].value for j in range(self.layer_sizes[1])] for i in range(self.layer_sizes[2])])) / 2
    self.b2_consensus = (np.array([self.model1.b2[i].value for i in range(self.layer_sizes[2])]) + 
                            np.array([self.model2.b2[i].value for i in range(self.layer_sizes[2])])) / 2

    
admm_penalty = (self.rho / 2) * (sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j])**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                                    sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i])**2 for i in range(self.layer_sizes[1])) +
                                    sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j])**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                                    sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i])**2 for i in range(self.layer_sizes[2])))

def admm_solve(self, iterations=50):
    for iter in range(iterations):
        print(f"ADMM Iteration {iter+1}/{iterations}")

        # Update model1 while keeping model2 fixed
        self.solve_model()
        self.update_consensus_variables()
        self.update_dual_variables()

        # Update model2 while keeping model1 fixed
        self.solve_model()
        self.update_consensus_variables()
        self.update_dual_variables()
        
        

## ======================================================================================================================================================== ##

class NeuralODEPyomoADMM:
    # ... existing code ...
    
    def update_objective(self):
        def _objective1(m):
            # data fit term
            data_fit = sum((m.y[i] - self.y_observed1[i])**2 for i in range(len(self.t1)))
            reg_smooth = sum((m.y[i] - m.y[i + 1])**2 for i in range(len(self.t1) - 1))

            # regularization term for weights and biases
            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) +
                   sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) +
                   sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) +
                   sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

            # ADMM penalty term for consensus
            admm_penalty = (self.rho / 2) * (
                sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j] / self.rho)**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i] / self.rho)**2 for i in range(self.layer_sizes[1])) +
                sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j] / self.rho)**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i] / self.rho)**2 for i in range(self.layer_sizes[2]))
            )

            return data_fit + self.penalty_lambda_reg * reg + self.penalty_lambda_smooth * reg_smooth + admm_penalty

        def _objective2(m):
            data_fit = sum((m.y[i] - self.y_observed2[i])**2 for i in range(len(self.t2)))
            reg_smooth = sum((m.y[i] - m.y[i + 1])**2 for i in range(len(self.t2) - 1))

            # regularization for weights and biases
            reg = (sum(m.W1[j, k]**2 for j in range(self.layer_sizes[1]) for k in range(self.layer_sizes[0])) +
                   sum(m.W2[j, k]**2 for j in range(self.layer_sizes[2]) for k in range(self.layer_sizes[1])) +
                   sum(m.b1[j]**2 for j in range(self.layer_sizes[1])) +
                   sum(m.b2[j]**2 for j in range(self.layer_sizes[2])))

            # ADMM penalty term for consensus
            admm_penalty = (self.rho / 2) * (
                sum((m.W1[i, j] - self.W1_consensus[i, j] + self.dual_W1[i, j] / self.rho)**2 for i in range(self.layer_sizes[1]) for j in range(self.layer_sizes[0])) +
                sum((m.b1[i] - self.b1_consensus[i] + self.dual_b1[i] / self.rho)**2 for i in range(self.layer_sizes[1])) +
                sum((m.W2[i, j] - self.W2_consensus[i, j] + self.dual_W2[i, j] / self.rho)**2 for i in range(self.layer_sizes[2]) for j in range(self.layer_sizes[1])) +
                sum((m.b2[i] - self.b2_consensus[i] + self.dual_b2[i] / self.rho)**2 for i in range(self.layer_sizes[2]))
            )

            return data_fit + self.penalty_lambda_reg * reg + self.penalty_lambda_smooth * reg_smooth + admm_penalty

        self.model1.obj = Objective(rule=_objective1, sense=pyo.minimize)
        self.model2.obj = Objective(rule=_objective2, sense=pyo.minimize)
