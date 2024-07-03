import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.experimental.ode import odeint
import optax

from flax import linen as nn
from flax.training import train_state
from jax import random
import flax.linen.initializers as initializers

class NeuralODE(nn.Module):
    layer_widths: list
    time_invariant: bool = True

    @nn.compact
    def __call__(self, x):
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width, kernel_init=initializers.lecun_normal())(x)
            # batch normalization
            # x = nn.BatchNorm(use_running_average=False)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1], kernel_init=initializers.lecun_normal())(x)
        return x

    def create_train_state(self, rng, learning_rate):
        """
        Create and initialize the training state for the NeuralODE model.
        
        Args:
            rng (jax.random.PRNGKey): Random number generator key.
            learning_rate (float): Learning rate for the optimizer.
        
        Returns:
            train_state.TrainState: Initialized training state.
        """
        params = self.init(rng, jnp.ones((self.layer_widths[0],)))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def loss_fn(self, params, apply_fn, t, observed_data, y0):
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
            y = jnp.atleast_1d(y)
            if self.time_invariant:
                return apply_fn({'params': params}, y)
            else:
                # time-dependent ODE
                input = jnp.append(y, t)
                return apply_fn({'params': params}, input)
        
        pred_solution = odeint(func, y0, t)
        # >>>>>>>>>>>>>>>>>>>>>>>>>> MSE vs MAE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        loss_mse = jnp.mean(jnp.square(pred_solution - observed_data))
        loss_mae = jnp.mean(jnp.abs(pred_solution - observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))  # L2 regularization
        return loss_mse + 1e-4 * l2_regularization 
    
    def train_step(self, state, t, observed_data, y0):
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
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(state.params, state.apply_fn, t, observed_data, y0)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, t, observed_data, y0, num_epochs=1000):
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
        @jax.jit
        def train_step_jit(state, t, observed_data, y0):
            return self.train_step(state, t, observed_data, y0)

        for epoch in range(num_epochs):
            state, loss = train_step_jit(state, t, observed_data, y0)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        return state

    def neural_ode(self, params, y0, t, state):
        """
        Obtain the solution of the neural ODE given initial conditions and time points.
        
        Args:
            params (dict): Parameters of the model.
            y0 (jax.numpy.ndarray): Initial condition for the ODE.
            t (jax.numpy.ndarray): Time points at which the ODE is solved.
            state (train_state.TrainState): State of the trained model.
        
        Returns:
            jax.numpy.ndarray: Solution of the ODE at the given time points.
        """
        def func(y, t):
            y = jnp.atleast_1d(y)
            if self.time_invariant:
                return state.apply_fn({'params': params}, y)
            else:
                input = jnp.append(y, t)
                return state.apply_fn({'params': params}, input)
            
        return odeint(func, y0, t)
