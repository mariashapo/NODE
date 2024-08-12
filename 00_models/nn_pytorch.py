import torch
import torch.nn as nn
from torchdiffeq import odeint  # Using regular odeint
from torch.optim import Adam

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d

import warnings

class NeuralODE(nn.Module):
    def __init__(self, layer_widths, learning_rate, time_invariant = True):
        super(NeuralODE, self).__init__()
        # construct the neural network layers
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
                
        self.network = nn.Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()
        self.time_invariant = time_invariant
                
    def create_interpolators(self):
        self.extra_intputs_checks()
        
        # initialize a list of interpolators
        self.interpolators = []
    
        # extra dimensions were removed so use ndim instead of shape
        if self.extra_inputs.ndim == 1: # 1D case
            f = interp1d(self.t, self.extra_inputs, kind='cubic', fill_value='extrapolate')
            self.interpolators.append(f)
        elif self.extra_inputs.ndim == 2: # 2D case
            for i in range(self.extra_inputs.shape[1]):
                # interpolate each column
                f = interp1d(self.t, np.squeeze(self.extra_inputs[:, i]), kind='cubic', fill_value='extrapolate')
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
                interpolated_input = interp(t.item())  
                interpolated_tensor = torch.tensor(interpolated_input, dtype=torch.float32).unsqueeze(-1)
                y = torch.cat((y, interpolated_tensor), dim=-1)
                
        return self.network(y)

    def train_model(self, t, observed_data, y0, num_epochs=1000, extra_inputs=None, rtol=1e-3, atol=1e-4):
        NeuralODE.ensure_tensor(t)
        NeuralODE.ensure_tensor(y0)
        NeuralODE.ensure_tensor(observed_data)
        
        self.extra_inputs = extra_inputs
        if self.extra_inputs is not None:
            self.extra_inputs = np.squeeze(extra_inputs) # remove extra dimensions
            # note: t will be coming in as a tensor
            self.t = t.clone().detach().numpy()
            self.create_interpolators()
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            pred_solution = odeint(self, y0, t, rtol=rtol, atol=atol)
            loss = self.criterion(pred_solution, observed_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    def predict(self, t, y0, extra_inputs = None):
        NeuralODE.ensure_tensor(t)
        NeuralODE.ensure_tensor(y0)
            
        with torch.no_grad():
            solution = odeint(self, y0, t)
        return solution
    
    def extra_intputs_checks(self):
        if self.extra_inputs.ndim == 1 and len(self.extra_inputs) != len(self.t):
            warnings.warn("extra_inputs has a different number of time points than the observed data")
        if self.extra_inputs.ndim == 2 and self.extra_inputs.shape[0] != len(self.t):
            warnings.warn("extra_inputs has a different number of time points than the observed data")
    
    @staticmethod
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x
    