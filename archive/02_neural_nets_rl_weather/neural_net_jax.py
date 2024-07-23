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
    loss: int = 0
    max_iter: int = np.inf
    regularizer: float = 1e-5

    @nn.compact
    def __call__(self, x):
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width, kernel_init=initializers.lecun_normal())(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1], kernel_init=initializers.lecun_normal())(x)
        return x

    def create_train_state(self, rng, learning_rate, regularizer = 1e-5):
        """
        Create and initialize the training state for the NeuralODE model.
        
        Args:
            rng (jax.random.PRNGKey): Random number generator key.
            learning_rate (float): Learning rate for the optimizer.
        
        Returns:
            train_state.TrainState: Initialized training state.
        """
        self.regularizer = regularizer
        
        params = self.init(rng, jnp.ones((self.layer_widths[0],)))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def loss_fn(self, params, apply_fn, t, observed_data, y0, args):
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
        def func(y, t, args_):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                # time-dependent ODE
                # print(f't shape: {t.shape}')
                input = jnp.append(input, t)
            
            # print(f'Input shape: {input.shape}') 
            if args is not None:
                # print(f'Args shape: {args.shape}')
                for arg in args[t.astype(int)]: # need to select the correct index
                    # print(f'Arg shape: {arg.shape}')
                    input = jnp.append(input, arg)
            
            # print(f'Input shape: {input.shape}')
            return apply_fn({'params': params}, input)
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>> ODEINT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< # 
        pred_solution = odeint(func, y0, t, args)
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>> MSE vs MAE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        loss_mse = jnp.mean(jnp.square(pred_solution - observed_data))
        # loss_mae = jnp.mean(jnp.abs(pred_solution - observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))  # L2 regularization
        
        return loss_mse + 1e-5 * l2_regularization 
    
    def train_step(self, state, t, observed_data, y0, extra_args):
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
        loss, grads = grad_fn(state.params, state.apply_fn, t, observed_data, y0, extra_args)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, t, observed_data, y0, num_epochs = np.inf, loss = 0, extra_args=None):
        self.loss = loss
        self.max_iter = num_epochs
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
        def train_step_jit(state, t, observed_data, y0, extra_args):
            return self.train_step(state, t, observed_data, y0, extra_args)

        epoch = 0
        # for epoch in range(num_epochs):
        while True:
            epoch += 1
            state, loss = train_step_jit(state, t, observed_data, y0, extra_args)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            if loss < self.loss or epoch > self.max_iter:
                break
        return state

    def neural_ode(self, params, y0, t, state, args = None):
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
        def func(y, t, args_):
            input = jnp.atleast_1d(y)
            
            if not self.time_invariant:
                input = jnp.append(input, t)
            
            if args_ is not None:
                print("*")
                # need to select the correct index based on the time point
                for arg in args_[t.astype(int)]: 
                    input = jnp.append(input, arg)
            
            return state.apply_fn({'params': params}, input)
            
        return odeint(func, y0, t, args)
