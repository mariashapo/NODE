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

    def create_train_state(self, rng, learning_rate, regularizer=1e-5):
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
            args (jax.numpy.ndarray): Extra arguments for the ODE function.
        
        Returns:
            float: The mean absolute error loss.
        """
        def func(y, t, args):
            input = jnp.atleast_1d(y)
            
            if not self.time_invariant:
                input = jnp.append(input, t)
            if args is not None:
                extra_inputs, t_all = args
                index = jnp.argmin(jnp.abs(t_all - t))
                
                for extra_input in extra_inputs[index]:
                        input = jnp.append(input, extra_input)
                        
                """if extra_inputs.shape[1] > 0:
                    for extra_input in extra_inputs[index]:
                        input = jnp.append(input, extra_input)
                else:
                    input = jnp.append(input, extra_inputs[index])"""
                
            return apply_fn({'params': params}, input)
        
        pred_solution = odeint(func, y0, t, args)
        
        loss_mse = jnp.sum(jnp.square(pred_solution - observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))
        
        return loss_mse #+ self.regularizer * l2_regularization 
    
    def train_step(self, state, t, observed_data, y0, extra_args):
        """
        Perform a single training step by computing the loss and its gradients,
        and then updating the model parameters.

        Args:
            state (train_state.TrainState): Contains model state including parameters.
            t (jax.numpy.ndarray): Time points at which the ODE is solved.
            observed_data (jax.numpy.ndarray): True data to compare against the model's predictions.
            y0 (jax.numpy.ndarray): Initial condition for the ODE.
            extra_args (jax.numpy.ndarray): Extra arguments for the ODE function.

        Returns:
            tuple: Updated state and loss value.
        """
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(state.params, state.apply_fn, t, observed_data, y0, extra_args)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, t, observed_data, y0, num_epochs=np.inf, loss=0, extra_args=None):
        """
        Train the model over a specified number of epochs.
        
        Args:
            state (train_state.TrainState): Initial state of the model.
            t (jax.numpy.ndarray): Time points at which the ODE is solved.
            observed_data (jax.numpy.ndarray): True data to compare against the model's predictions.
            y0 (jax.numpy.ndarray): Initial condition for the ODE.
            num_epochs (int, optional): Number of training epochs. Default is 1000.
            loss (float, optional): Loss threshold for stopping the training. Default is 0.
            extra_args (jax.numpy.ndarray, optional): Extra arguments for the ODE function.

        Returns:
            train_state.TrainState: Trained model state.
        """
        self.loss = loss
        self.max_iter = num_epochs
        
        @jax.jit
        def train_step_jit(state, t, observed_data, y0, extra_args):
            return self.train_step(state, t, observed_data, y0, extra_args)

        epoch = 0
        while True:
            epoch += 1
            state, loss = train_step_jit(state, t, observed_data, y0, extra_args)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            if loss < self.loss or epoch > self.max_iter:
                break
        return state

    def neural_ode(self, params, y0, t, state, extra_args=None):
        """
        Obtain the solution of the neural ODE given initial conditions and time points.
        
        Args:
            params (dict): Parameters of the model.
            y0 (jax.numpy.ndarray): Initial condition for the ODE.
            t (jax.numpy.ndarray): Time points at which the ODE is solved.
            state (train_state.TrainState): State of the trained model.
            args (jax.numpy.ndarray, optional): Extra arguments for the ODE function.

        Returns:
            jax.numpy.ndarray: Solution of the ODE at the given time points.
        """
        
        def func(y, t, args):
            print("XXX")
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)
            if args is not None:
                extra_inputs, t_all = args
                
                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    print(t_all.shape)
                    index = jnp.argmin(jnp.abs(t_all - t))
                    
                    if extra_inputs.shape[1] > 0:
                        for extra_input in extra_inputs[index]:
                            input = jnp.append(input, extra_input)
                    else:
                        input = jnp.append(input, extra_inputs[index])
                        
                else: # if a single value, simply append it
                    input = jnp.append(input, extra_inputs)
                        
            return state.apply_fn({'params': params}, input)
            
        return odeint(func, y0, t, extra_args)
