import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.experimental.ode import odeint
import optax

from flax import linen as nn
from flax.training import train_state
from jax import random


#--------------------------------------------ODEINT-------------------------------------------#
class NeuralODE(nn.Module):
    """
    A neural network model for solving ODEs using neural networks.
    
    Attributes:
        layer_widths (list): List of integers specifying the width of each layer in the neural network.
    """
    layer_widths: list

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (jax.numpy.ndarray): Input data.
        
        Returns:
            jax.numpy.ndarray: Output of the neural network.
        """
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1])(x)
        return x

def create_train_state(rng, layer_widths, learning_rate):
    """
    Create and initialize the training state for the NeuralODE model.
    
    Args:
        rng (jax.random.PRNGKey): Random number generator key.
        layer_widths (list): List of integers specifying the width of each layer in the neural network.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        train_state.TrainState: Initialized training state.
    """
    model = NeuralODE(layer_widths=layer_widths)
    params = model.init(rng, jnp.ones((2,)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def loss_fn(params, apply_fn, t, observed_data, y0):
    """
    Compute the loss as the mean absolute error between predicted and observed data.
    
    Args:
        params (dict): Parameters of the model.
        apply_fn (function): Function to apply the model to input data.
        t (jax.numpy.ndarray): Time points at which the ODE is solved.
        observed_data (jax.numpy.ndarray): True data to compare against the model's predictions.
        y0 (jax.numpy.ndarray): Initial condition for the ODE.
    
    Returns:
        float: The mean absolute error loss.
    """
    def func(y, t):
        return apply_fn({'params': params}, y)
    
    pred_solution = odeint(func, y0, t)
    return jnp.mean(jnp.abs(pred_solution - observed_data))

@jax.jit
def train_step(state, t, observed_data, y0):
    """
    Perform a single training step by computing the loss and its gradients,
    and then updating the model parameters.

    Args:
        state (train_state.TrainState): Contains model state including parameters.
        t (jax.numpy.ndarray): Time points at which the ODE is solved.
        observed_data (jax.numpy.ndarray): True data to compare against the model's predictions.
        y0 (jax.numpy.ndarray): Initial condition for the ODE.

    Returns:
        tuple: Updated state and loss value.
    """
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, t, observed_data, y0)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(state, t, observed_data, y0, num_epochs=1000):
    """
    Train the model over a specified number of epochs.
    
    Args:
        state (train_state.TrainState): Initial state of the model.
        t (jax.numpy.ndarray): Time points at which the ODE is solved.
        observed_data (jax.numpy.ndarray): True data to compare against the model's predictions.
        y0 (jax.numpy.ndarray): Initial condition for the ODE.
        num_epochs (int, optional): Number of training epochs. Default is 1000.
    
    Returns:
        train_state.TrainState: Trained model state.
    """
    for epoch in range(num_epochs):
        state, loss = train_step(state, t, observed_data, y0)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return state

# Helper function to obtain the solution
def neural_ode(params, y0, t, state):
    def func(y, t):
        return state.apply_fn({'params': params}, y)
    return odeint(func, y0, t)

#-----------------------------------------COLLOCATION-----------------------------------------#
class NeuralODE_Collocation(nn.Module):
    """
    A neural network model for approximating ODE solutions using collocation.
    
    Attributes:
        layer_widths (list): List of integers specifying the width of each layer in the neural network.
    """
    layer_widths: list

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x (jax.numpy.ndarray): Input data.
        
        Returns:
            jax.numpy.ndarray: Output of the neural network.
        """
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1])(x)
        return x
    
def create_train_state_collocation(rng, layer_widths, input_shape, learning_rate):
    """
    Create and initialize the training state for the model.
    
    Args:
        rng (jax.random.PRNGKey): Random number generator key.
        layer_widths (list): List of integers specifying the width of each layer in the neural network.
        input_shape (tuple): Shape of the input data.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        train_state.TrainState: Initialized training state.
    """
    model = NeuralODE_Collocation(layer_widths=layer_widths)
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def loss_fn_collocation(params, apply_fn, t, observed_derivative):
    """
    Compute the loss as the mean absolute error between predicted and observed derivatives.
    
    Args:
        params (dict): Parameters of the model.
        apply_fn (function): Function to apply the model to input data.
        t (jax.numpy.ndarray): Input features; typically, the time points or other relevant features.
        observed_derivative (jax.numpy.ndarray): True derivatives at the input features.
    
    Returns:
        float: The mean absolute error loss.
    """
    pred_derivative = apply_fn({'params': params}, t)
    return jnp.mean(jnp.abs(pred_derivative - observed_derivative))

@jit
def train_step_collocation(state, t, observed_derivative):
    """
    Perform a single training step by computing the loss and its gradients,
    and then updating the model parameters.

    Args:
        state (train_state.TrainState): Contains model state including parameters.
        t (jax.numpy.ndarray): Input features for the model.
        observed_derivative (jax.numpy.ndarray): True derivatives to compare against model predictions.

    Returns:
        tuple: Updated state and loss value.
    """
    def loss_fn(params):
        return loss_fn_collocation(params, state.apply_fn, t, observed_derivative)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train_collocation(state, t, observed_data, num_epochs=1000):
    """
    Train the model using the collocation method for a given number of epochs.
    
    Args:
        state (train_state.TrainState): Initial state of the model.
        t (jax.numpy.ndarray): Input features for the model.
        observed_data (jax.numpy.ndarray): True derivatives to compare against model predictions.
        num_epochs (int, optional): Number of training epochs. Default is 1000.
    
    Returns:
        train_state.TrainState: Trained model state.
    """
    for epoch in range(num_epochs):
        state, loss = train_step_collocation(state, t, observed_data)
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return state

def predict_derivatives(state, t):
    """
    Predict the derivatives using the trained model.
    
    Args:
        state (train_state.TrainState): Trained model state.
        t (jax.numpy.ndarray): Input features for the model.
    
    Returns:
        jax.numpy.ndarray: Predicted derivatives.
    """
    pred_dy_dt = []
    for i in range(len(t)):
        dy_dt = state.apply_fn({'params': state.params}, t[i])
        pred_dy_dt.append(dy_dt)
    return jnp.array(pred_dy_dt)
