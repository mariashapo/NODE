import torch
import torch.nn as nn
from torchdiffeq import odeint  # Using regular odeint
from torch.optim import Adam

import random
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d

import warnings

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)       # Python random module.
    np.random.seed(seed)    # Numpy module.
    torch.manual_seed(seed) # PyTorch to ensure reproducibility for CPU

    # if using CUDA (PyTorch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # Below two ensure consistency, but might slow down your code
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class NeuralODE(nn.Module):
    def __init__(self, layer_widths, learning_rate, weight_decay = 1e-5, custom_weights=None, time_invariant=True):
        super(NeuralODE, self).__init__()
        set_seed(42)
        layers = []
        self.layer_widths = layer_widths

        for i in range(len(layer_widths) - 1):
            layer = nn.Linear(layer_widths[i], layer_widths[i + 1])
            # Initialize weights if custom weights are provided
            if custom_weights and i < len(custom_weights):
                weight, bias = custom_weights[i]
                layer.weight.data = torch.from_numpy(weight).float()
                layer.bias.data = torch.from_numpy(bias).float()
            layers.append(layer)
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.MSELoss()
        self.time_invariant = time_invariant

                
    def create_interpolators(self):
        self.extra_intputs_checks()
        
        # initialize a list of interpolators
        self.interpolators = []
    
        # extra dimensions were removed so use ndim instead of shape
        if self.extra_inputs.ndim == 1: # 1D case
            f = interp1d(self.t_all, self.extra_inputs, kind='cubic', fill_value='extrapolate')
            self.interpolators.append(f)
        elif self.extra_inputs.ndim == 2: # 2D case
            for i in range(self.extra_inputs.shape[1]):
                # interpolate each column
                f = interp1d(self.t_all, np.squeeze(self.extra_inputs[:, i]), kind='cubic', fill_value='extrapolate')
                self.interpolators.append(f)
        else:
            raise ValueError("extra_inputs must be either 1D or 2D. Received {}D.".format(self.extra_inputs.ndim))
            
    def forward(self, t, y):
        if not self.time_invariant:
            # add a new dimension to make t (batch_size, 1)
            t = t.clone().detach().unsqueeze(-1) 
            y = torch.cat((y, t), dim=-1) 
            
        if self.extra_inputs is not None:
            for interp in self.interpolators:
                # convert tensor to a value and interpolate
                # from each interpolator extract the interpolated value at time t
                interpolated_input = interp(t.item())  
                interpolated_tensor = torch.tensor(interpolated_input, dtype=torch.float32).unsqueeze(-1)
                y = torch.cat((y, interpolated_tensor), dim=-1)
                
        return self.network(y)

    def train_model(self, t, observed_data, y0, num_epochs=1000, extra_inputs=None, 
                        rtol=1e-3, atol=1e-4,
                        termination_loss=0, verbose=True, log=None):
        
            y0 = NeuralODE.ensure_tensor(y0)
            t = NeuralODE.ensure_tensor(t)
            observed_data = NeuralODE.ensure_tensor(observed_data)

            self.extra_inputs = extra_inputs
            
            if self.extra_inputs is not None:
                self.t_all, extra_inputs = extra_inputs
                self.extra_inputs = np.squeeze(extra_inputs) # remove extra dimensions
                # note: t will be coming in as a tensor
                self.create_interpolators()

            losses_test = []  # To store test losses
            losses_recalculated = []  # To store recalculated training losses

            for epoch in range(num_epochs):
                self.train()
                self.optimizer.zero_grad()
                pred_solution = odeint(self, y0, t, rtol=rtol, atol=atol)
                loss = self.criterion(pred_solution, observed_data)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                # if logging is enabled, calculate the loss at the specified interval
                if log and (epoch % log['epoch_recording_step'] == 0 or epoch == num_epochs - 1):
                    with torch.no_grad():
                        self.eval()
                        # Recalculate the training loss using odeint
                        # log['y_init'], log['t']
                        recalculated_pred = odeint(self, log['y_init'], log['t'])
                        recalculated_loss = np.mean(np.square(recalculated_pred.detach().numpy() - log['y'].detach().numpy()))
                        losses_recalculated.append(recalculated_loss)

                        # Calculate test loss if test data is provided
                        if 'y_test' in log and 't_test' in log:
                            pred_test = odeint(self, log['y_init_test'], log['t_test'])
                            test_loss = np.mean(np.square(pred_test.detach().numpy() - log['y_test'].detach().numpy()))
                            losses_test.append(test_loss)
                        
                    self.train()

                if verbose and epoch % 100 == 0:
                    print(f'Epoch {epoch}, Training Loss: {loss.item()}')

                '''if loss.item() < termination_loss:
                    print(f"Early stopping at epoch {epoch} with training loss {loss.item()}")
                    break'''

            self.losses = (losses_recalculated, losses_test)
            return self.losses
    
    def predict(self, t, y0, extra_inputs = None):
        NeuralODE.ensure_tensor(t)
        NeuralODE.ensure_tensor(y0)
            
        with torch.no_grad():
            solution = odeint(self, y0, t)
        return solution
    
    def extra_intputs_checks(self):
        if self.extra_inputs.ndim == 1 and len(self.extra_inputs) != len(self.t_all):
            warnings.warn("extra_inputs has a different number of time points than the observed data")
        if self.extra_inputs.ndim == 2 and self.extra_inputs.shape[0] != len(self.t_all):
            warnings.warn("extra_inputs has a different number of time points than the observed data")
    
    @staticmethod
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x
    