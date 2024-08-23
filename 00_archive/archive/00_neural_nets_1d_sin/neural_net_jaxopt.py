import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental.ode import odeint
from jaxopt import ScipyMinimize

"""
    ::: Work in progress :::
    * Ensure both 1d and 2d inputs work
    * For now only 2d, u and v, inputs are supported 
"""

class ODEOptimizationModelJaxopt:
    def __init__(self, y_observed, t, D, 
                 layer_sizes, penalty_lambda=2, smoothing_lambda=100, 
                 max_iter=1000, act_func="tanh", w_init_method="random", random_seed=0,
                 extra_inputs = None):
        self.y_observed = y_observed
        self.t = t
        self.D = D
        self.penalty_lambda = penalty_lambda
        self.smoothing_lambda = smoothing_lambda
        self.observed_dim = None
        self.max_iter = max_iter
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.rng = jax.random.PRNGKey(random_seed)  # Allow different random seeds

    def initialize_params(self, u_init, v_init):
        params = {}
        input_size = self.layer_sizes[0]
        for i in range(len(self.layer_sizes) - 1):
            self.rng, subkey = jax.random.split(self.rng)
            output_size = self.layer_sizes[i + 1]
            params[f'W{i}'] = self.initialize_weights(subkey, (output_size, input_size))
            params[f'b{i}'] = self.initialize_biases(subkey, output_size)
            input_size = output_size
        # Initialize u and v as variables to be optimized
        params['u'] = u_init
        params['v'] = v_init
        return params

    def initialize_weights(self, rng, shape):
        if self.w_init_method == 'random':
            return jax.random.normal(rng, shape) * 0.1
        elif self.w_init_method == 'xavier':
            return jax.random.normal(rng, shape) * jnp.sqrt(2 / (shape[0] + shape[1]))
        elif self.w_init_method == 'he':
            return jax.random.normal(rng, shape) * jnp.sqrt(2 / shape[0])
        else:
            raise ValueError("Unsupported initialization method. Use 'random', 'xavier', or 'he'.")

    def initialize_biases(self, rng, size):
        return jax.random.normal(rng, (size,)) * 0.1

    def nn_output(self, params, t, u, v):
        inputs = jnp.array([u, v])
        for i in range(len(self.layer_sizes) - 1):
            W = params[f'W{i}']
            b = params[f'b{i}']
            inputs = jnp.dot(W, inputs) + b
            if i < len(self.layer_sizes) - 2:
                if self.act_func == "tanh":
                    inputs = jnp.tanh(inputs)
                elif self.act_func == "sigmoid":
                    inputs = 1 / (1 + jnp.exp(-inputs))
                elif self.act_func == "softplus":
                    inputs = jax.nn.softplus(inputs)
        return inputs

    def loss_fn(self, params, t, y_observed, D):
        
        def collocation_residual(params, i):
            du_dt = jnp.dot(D[i], params['u'])
            dv_dt = jnp.dot(D[i], params['v'])
            nn_u, nn_v = self.nn_output(params, t[i], params['u'][i], params['v'][i])
            return ((nn_u - du_dt)**2 + (nn_v - dv_dt)**2) / 2

        u_pred = params['u']
        v_pred = params['v']

        # model.u, model.v - y_observed
        data_fit = jnp.sum((jnp.stack((u_pred, v_pred), axis=-1) - y_observed)**2)
        
        # collocation_residual
        penalty_terms = jax.vmap(lambda i: collocation_residual(params, i))(jnp.arange(1, len(t)))
        penalty = jnp.sum(penalty_terms)

        # smoothing term
        second_derivative = jnp.sum(jnp.diff(jnp.stack((u_pred, v_pred), axis=-1), n=2, axis=0)**2, axis=0)
        smoothing_term = jnp.sum(second_derivative)

        return data_fit + self.penalty_lambda * penalty + self.smoothing_lambda * smoothing_term

    def optimize_model(self, params):
        solver = ScipyMinimize(fun=self.loss_fn, method="BFGS", maxiter=self.max_iter)
        opt_result = solver.run(params, t=self.t, y_observed=self.y_observed, D=self.D)
        # Return optimized parameters and loss value
        return opt_result.params, opt_result.state.fun_val

    def predict(self, params, t, initial_conditions):
        
        def model(state, t):
            u, v = state
            nn_u, nn_v = self.nn_output(params, t, u, v)
            return nn_u, nn_v

        states = odeint(model, initial_conditions, t)
        u_pred, v_pred = states[:, 0], states[:, 1]
        return u_pred, v_pred
    
    def extract_uv_params(self, params):
        return params['u'], params['v']


# if __name__ == "__main__":
    
