import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.optim import Adam

import numpy as np
import warnings

class NeuralODE(nn.Module):
    def __init__(self, layer_widths, learning_rate, time_invariant=True):
        super(NeuralODE, self).__init__()
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
                
        self.network = nn.Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.time_invariant = time_invariant

    def create_interpolators(self, t, extra_inputs):
        self.t_grid = torch.tensor(t, dtype=torch.float32)
        self.extra_inputs = torch.tensor(extra_inputs, dtype=torch.float32)
        
    def interpolate(self, t):
        # Linear interpolation in PyTorch
        idx = torch.searchsorted(self.t_grid, t) - 1
        idx = idx.clamp(min=0, max=len(self.t_grid) - 2)
        weight = (t - self.t_grid[idx]) / (self.t_grid[idx + 1] - self.t_grid[idx])
        
        interpolated = self.extra_inputs[idx] * (1 - weight) + self.extra_inputs[idx + 1] * weight
        return interpolated

    def forward(self, t, y):
        if not self.time_invariant:
            t = t.clone().detach().unsqueeze(-1)
            y = torch.cat((y, t), dim=-1)
            
        if hasattr(self, 'extra_inputs'):
            interpolated_input = self.interpolate(t)
            y = torch.cat((y, interpolated_input.unsqueeze(-1)), dim=-1)
        
        return self.network(y)

    def train_model(self, t, observed_data, y0, num_epochs=1000, extra_inputs=None):
        NeuralODE.ensure_tensor(t)
        NeuralODE.ensure_tensor(y0)
        
        if extra_inputs is not None:
            self.create_interpolators(t.numpy(), np.squeeze(extra_inputs))
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            pred_solution = odeint(self, y0, t)
            loss = self.criterion(pred_solution, observed_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    def predict(self, t, y0):
        NeuralODE.ensure_tensor(t)
        NeuralODE.ensure_tensor(y0)
            
        with torch.no_grad():
            solution = odeint(self, y0, t)
        return solution

    @staticmethod
    def ensure_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x
