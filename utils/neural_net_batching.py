import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random
import flax.linen.initializers as initializers
from jax import value_and_grad

class NeuralODE(nn.Module):
    layer_widths: list
    time_invariant: bool = True

    @nn.compact
    def __call__(self, x):
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width, kernel_init=initializers.lecun_normal())(x)
            # x = nn.BatchNorm(use_running_average=False)(x)  # Uncomment if batch normalization is needed
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1], kernel_init=initializers.lecun_normal())(x)
        return x

    def create_train_state(self, rng, learning_rate):
        params = self.init(rng, jnp.ones((self.layer_widths[0],)))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def loss_fn(self, params, apply_fn, batch_t, batch_observed_data, y0):
        # batch_size = batch_observed_data.shape[0]

        def func(y, t):
            y = jnp.atleast_1d(y)
            if self.time_invariant:
                return apply_fn({'params': params}, y)
            else:
                input = jnp.append(y, t)
                return apply_fn({'params': params}, input)

        pred_solution = jax.vmap(lambda y0_single: odeint(func, y0_single, batch_t))(y0)

        # ensure the shapes match
        loss_mse = jnp.mean(jnp.square(pred_solution - batch_observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))  # L2 regularization
        return loss_mse + 1e-4 * l2_regularization

    def train_step(self, state, batch_t, batch_observed_data, y0):
        grad_fn = value_and_grad(self.loss_fn)
        loss, grads = grad_fn(state.params, state.apply_fn, batch_t, batch_observed_data, y0)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, t, observed_data, y0, batch_size, num_epochs=1000):
        batch_data = self.create_batches_with_unique_times(observed_data, y0, t, batch_size)
        num_batches = len(batch_data)

        @jax.jit
        def train_step_jit(state, batch_t, observed_data_batch, y0_batch):
            return self.train_step(state, batch_t, observed_data_batch, y0_batch)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_t, batch_observed_data, batch_y0 in batch_data:
                state, loss = train_step_jit(state, batch_t, batch_observed_data, batch_y0)
                epoch_loss += loss

            epoch_loss /= num_batches

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Average Loss: {epoch_loss}')

        return state

    def neural_ode(self, params, y0, t, state):
        def func(y, t):
            y = jnp.atleast_1d(y)
            if self.time_invariant:
                return state.apply_fn({'params': params}, y)
            else:
                input = jnp.append(y, t)
                return state.apply_fn({'params': params}, input)
            
        return odeint(func, y0, t)

    def create_batches_with_unique_times(self, observed_data, y0, t, batch_size):
        dataset_size = observed_data.shape[0]
        num_batches = dataset_size // batch_size
        batch_data = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_y0 = observed_data[batch_start]
            batch_observed_data = observed_data[batch_start:batch_end]
            batch_t = t[batch_start:batch_end]
            print(f"Batch t: {batch_t.shape}, batch_observed_data: {batch_observed_data.shape}, batch_y0: {batch_y0.shape}")
            batch_data.append((batch_t, batch_observed_data, batch_y0))
        return batch_data

    
# Example usage:
