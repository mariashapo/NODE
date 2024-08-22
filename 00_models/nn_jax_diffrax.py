import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.experimental import host_callback
import optax

from flax import linen as nn
from flax.training import train_state
from jax import random
import flax.linen.initializers as initializers
import diffrax as dfx

class NeuralODE(nn.Module):
    layer_widths: list
    time_invariant: bool = True
    loss: int = 0
    max_iter: int = np.inf
    regularizer: float = 1e-5
    act_func: callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for width in self.layer_widths[:-1]:
            x = nn.Dense(width, kernel_init=initializers.lecun_normal())(x)
            x = self.act_func(x)
        x = nn.Dense(self.layer_widths[-1], kernel_init=initializers.lecun_normal())(x)
        return x

    def create_train_state(self, rng, learning_rate, regularizer=1e-5, rtol = 1e-3, atol = 1e-6, dt0 = 1e-3):
        self.regularizer = regularizer
        self.rtol = rtol
        self.atol = atol
        self.dt0 = dt0
        
        params = self.init(rng, jnp.ones((self.layer_widths[0],)))['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)

    def loss_fn(self, params, apply_fn, t, observed_data, y0, args):
        # func acts as a forward pass for the neural ODE
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)
            if args is not None:
                extra_inputs, t_all = args

                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    # interpolate extra_inputs to get continuous values
                    if extra_inputs.ndim == 2:
                        interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(t, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, extra_inputs)

            result = apply_fn({'params': params}, input)
            return result
        
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            dfx.ODETerm(func),
            solver,
            t0=t[0],
            t1=t[-1],
            dt0 = self.dt0,
            y0=y0,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
            #, adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=100) 
        )

        pred_solution = solution.ys
        loss_mse = jnp.mean(jnp.square(pred_solution - observed_data))
        l2_regularization = sum(jnp.sum(param ** 2) for param in jax.tree_util.tree_leaves(params))
        
        return loss_mse + self.regularizer * l2_regularization

    def train_step(self, state, t, observed_data, y0, extra_args):
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(state.params, state.apply_fn, t, observed_data, y0, extra_args)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, state, t, observed_data, y0, num_epochs=np.inf, termination_loss=0, extra_args=None, verbose=True, 
              log=False):
        self.term_loss = termination_loss
        self.max_iter = num_epochs
        
        @jax.jit
        def train_step_jit(state, t, observed_data, y0, extra_args):
            return self.train_step(state, t, observed_data, y0, extra_args)

        losses = []
        
        epoch = 0
        while True:
            epoch += 1
            state, loss = train_step_jit(state, t, observed_data, y0, extra_args)
                      
            if log and epoch % 10 == 0:
                if jnp.squeeze(observed_data).shape[0] != log['t'].shape[0]:
                    k = jnp.squeeze(observed_data).shape[0]
                    pred = self.neural_ode(state.params, log['y_init'], log['t'][:k], state)
                    losses.append(jnp.mean(jnp.square(pred - log['y'][:k])))
                else:
                    pred = self.neural_ode(state.params, log['y_init'], log['t'], state)
                    losses.append(np.mean(np.square(pred - log['y'])))

            if epoch % 100 == 0:
                if verbose:
                    print(f'Epoch {epoch}, Loss: {loss}')
            if loss < self.term_loss or epoch > self.max_iter:
                break
            
        return state, losses

    def neural_ode(self, params, y0, t, state, extra_args=None): 
        def func(t, y, args):
            input = jnp.atleast_1d(y)
            if not self.time_invariant:
                input = jnp.append(input, t)

            if args is not None:
                extra_inputs, t_all = args
                if isinstance(extra_inputs, (np.ndarray, jnp.ndarray)):
                    # interpolate extra_inputs to get continuous values
                    if extra_inputs.ndim == 2:
                        interpolated_inputs = jnp.array([jnp.interp(t, t_all, extra_inputs[:, i]) for i in range(extra_inputs.shape[1])])
                        input = jnp.append(input, interpolated_inputs)
                    elif extra_inputs.ndim == 1:
                        interpolated_input = jnp.interp(t, t_all, extra_inputs)
                        input = jnp.append(input, interpolated_input)
                else:
                    input = jnp.append(input, extra_inputs)

            result = state.apply_fn({'params': params}, input)
            return result
        
        term = dfx.ODETerm(func)
        solver = dfx.Tsit5()
        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)
        saveat = dfx.SaveAt(ts=t)

        solution = dfx.diffeqsolve(
            term, # function
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=1e-3,
            y0=y0,
            args=extra_args,
            stepsize_controller=stepsize_controller,
            saveat=saveat
        )
        return solution.ys

def debug_print_simple(value):
    """A function to print during JIT execution using host_callback.id_tap."""
    def print_func(x, _):
        print(x)
        return x  
    return host_callback.id_tap(print_func, value)


def debug_print(value, transform=lambda x: x):
    """A function to print during JIT execution using host_callback.id_tap."""
    def print_func(x, _):
        print(transform(x))
        return x  
    return host_callback.id_tap(print_func, value)


