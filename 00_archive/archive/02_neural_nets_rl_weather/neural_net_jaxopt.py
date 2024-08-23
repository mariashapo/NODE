import jax
import jax.numpy as jnp
from jaxopt import GradientDescent, objective
import warnings

class NeuralODEJaxOpt:
    def __init__(self, y_observed, t, first_derivative_matrix, layer_sizes, time_invariant=True, extra_input=None, penalty_lambda=1, max_iter=500, act_func="tanh", w_init_method="random", params=None, y_init=None):
        self.y_observed = y_observed
        self.t = t
        self.first_derivative_matrix = first_derivative_matrix
        self.penalty_lambda = penalty_lambda
        self.max_iter = max_iter
        self.act_func = act_func
        self.w_init_method = w_init_method
        self.layer_sizes = layer_sizes
        self.y_init = y_init
        self.time_invariant = time_invariant
        self.extra_inputs = extra_input
        self.params = params

        # Initialize weights and biases
        self.weights, self.biases = self.initialize_network()

    def initialize_network(self):
        weights = []
        biases = []
        layer_dims = [self.layer_sizes[0]] + self.layer_sizes[1:]

        for i in range(len(layer_dims) - 1):
            weight_shape = (layer_dims[i+1], layer_dims[i])
            if self.w_init_method == 'random':
                weights.append(jnp.random.normal(size=weight_shape) * 0.1)
            elif self.w_init_method == 'xavier':
                weights.append(jnp.random.normal(size=weight_shape) * jnp.sqrt(2 / (layer_dims[i] + layer_dims[i+1])))
            elif self.w_init_method == 'he':
                weights.append(jnp.random.normal(size=weight_shape) * jnp.sqrt(2 / layer_dims[i]))
            else:
                raise ValueError("Unsupported initialization method. Use 'random', 'xavier', or 'he'.")

            biases.append(jnp.zeros(layer_dims[i+1]))

        return weights, biases

    def neural_net(self, inputs, weights, biases):
        activations = inputs

        for w, b in zip(weights[:-1], biases[:-1]):
            activations = jnp.dot(w, activations) + b
            if self.act_func == "tanh":
                activations = jnp.tanh(activations)
            elif self.act_func == "sigmoid":
                activations = jax.nn.sigmoid(activations)
            elif self.act_func == "softplus":
                activations = jax.nn.softplus(activations)

        # Output layer
        outputs = jnp.dot(weights[-1], activations) + biases[-1]
        return outputs

    def loss_function(self, weights, biases):
        mse_loss = 0
        for i in range(1, self.y_observed.shape[0]):
            dy_dt = jnp.dot(self.first_derivative_matrix[i], self.y_observed[i])
            nn_input = self.y_observed[i]
            nn_output = self.neural_net(nn_input, weights, biases)
            mse_loss += jnp.mean((nn_output - dy_dt) ** 2)

        return mse_loss

    def train_model(self):
        solver = GradientDescent(fun=self.loss_function, maxiter=self.max_iter)
        result = solver.run((self.weights, self.biases))
        self.weights, self.biases = result.params

    def predict(self, input_data):
        return self.neural_net(input_data, self.weights, self.biases)

    def mse(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    # Example usage
    # neural_ode = NeuralODEJaxOpt(y_observed, t, first_derivative_matrix, layer_sizes)
    # neural_ode.train_model()
    # predictions = neural_ode.predict(new_inputs)
    pass
