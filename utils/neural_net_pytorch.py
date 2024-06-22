import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Define a PyTorch module for the Neural ODE
class NeuralODE_Py(nn.Module):
    def __init__(self, layer_widths):
        super(NeuralODE_Py, self).__init__()
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, t, y):
        return self.network(y)

def create_train_state_py(layer_widths, learning_rate):
    model = NeuralODE_Py(layer_widths=layer_widths)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

def ode_system(model, y0, t):
    # t needs to be a tensor and y0 should be of appropriate shape
    t = torch.tensor(t, dtype=torch.float32)
    y0 = torch.tensor(y0, dtype=torch.float32)
    solution = odeint(model, y0, t)
    return solution

def train_py(model, optimizer, t, observed_data, y0, num_epochs=1000):
    criterion = torch.nn.MSELoss()  # Using MSE for simplicity
    t = torch.tensor(t, dtype=torch.float32)
    observed_data = observed_data.clone().detach()
    y0 = torch.tensor(y0, dtype=torch.float32)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred_solution = odeint(model, y0, t)
        loss = criterion(pred_solution, observed_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return model

def neural_ode_py(model, y0, t):
    """
    Use the trained neural ODE model to solve the ODE from initial condition y0 over time points t.

    Args:
        model (torch.nn.Module): The trained Neural ODE model.
        y0 (torch.Tensor or np.ndarray): Initial condition of the ODE.
        t (np.ndarray): Array of time points at which to solve the ODE.

    Returns:
        torch.Tensor: The solution of the ODE at the specified time points.
    """
    if not isinstance(y0, torch.Tensor):
        y0 = torch.tensor(y0, dtype=torch.float32)
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    with torch.no_grad():  # Use no_grad() to prevent tracking history during inference
        solution = odeint(model, y0, t)
    return solution
