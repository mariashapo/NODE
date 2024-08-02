import torch
import torch.nn as nn
from torchdiffeq import odeint  # Using regular odeint
from torch.optim import Adam

class NeuralODE(nn.Module):
    def __init__(self, layer_widths, learning_rate):
        super(NeuralODE, self).__init__()
        # Construct the neural network layers
        layers = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def forward(self, t, y):
        return self.network(y)

    def train_model(self, t, observed_data, y0, num_epochs=1000):
        observed_data = observed_data.clone().detach()
        
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
        if not isinstance(y0, torch.Tensor):
            y0 = torch.tensor(y0, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
            
        with torch.no_grad():
            solution =  odeint(self, y0, t)
        return solution