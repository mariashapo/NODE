import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state
import diffrax

# Define the neural network for the Neural ODE
class NeuralODE(nn.Module):
    layer_widths: list
    time_invariant: bool = True

    @nn.compact
    def __call__(self, x):
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width, kernel_init=nn.initializers.lecun_normal())(x)
            x = nn.tanh(x)
        x = nn.Dense(self.layer_widths[-1], kernel_init=nn.initializers.lecun_normal())(x)
        return x

    def create_train_state(self, rng, learning_rate):
        params = self.init(rng, jnp.ones((self.layer_widths[0],)))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def loss_fn(self, params, apply_fn, ts, observed_data, y0):
        def func(t, y, args):
            if self.time_invariant:
                return apply_fn({'params': params}, y)
            else:
                input = jnp.append(y, t)
                return apply_fn({'params': params}, input)
        
        term = diffrax.ODETerm(func)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0], y0=y0, saveat=diffrax.SaveAt(ts=ts))
        
        pred_solution = sol.ys
        loss = jnp.mean(jnp.square(pred_solution - observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))
        return loss + 1e-4 * l2_regularization

    def train_step(self, state, ts, observed_data, y0):
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(state.params, state.apply_fn, ts, observed_data, y0)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, ts, observed_data, y0, num_epochs=1000):
        for epoch in range(num_epochs):
            state, loss = self.train_step(state, ts, observed_data, y0)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        return state

    def neural_ode(self, params, y0, ts, state):
        def func(t, y, args):
            if self.time_invariant:
                return state.apply_fn({'params': params}, y)
            else:
                input = jnp.append(y, t)
                return state.apply_fn({'params': params}, input)
            
        term = diffrax.ODETerm(func)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0], y0=y0, saveat=diffrax.SaveAt(ts=ts))
        return sol.ys

   
