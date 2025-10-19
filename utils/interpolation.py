import jax
import jax.numpy as jnp

class BarycentricInterpolation:
    def __init__(self, n, start=-5, stop=5, spacing = "Chebyshev"):
        self.n = n
        self.start = start
        self.stop = stop
        self.nodes = self.chebyshev_nodes_second_kind()
        self.weights = self.compute_barycentric_weights()
    
        self.coefficients = None

    def chebyshev_nodes_second_kind(self):
        k = jnp.arange(self.n)
        x = jnp.cos(jnp.pi * k / (self.n - 1))
        nodes = 0.5 * (self.stop - self.start) * x + 0.5 * (self.start + self.stop)
        return jnp.sort(nodes)

    def compute_barycentric_weights(self):
        w = jnp.ones(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    w = w.at[i].set(w[i] / (self.nodes[i] - self.nodes[j]))
        return w

    def fit(self, y):
        if len(y) != self.n:
            raise ValueError("The length of y must match the number of nodes.")
        self.coefficients = jnp.array(y)

    def evaluate(self, x):
        x = jnp.asarray(x)
        numerator = jnp.zeros_like(x, dtype=float)
        denominator = jnp.zeros_like(x, dtype=float)
        exact_matches = jnp.isclose(x[:, None], self.nodes)

        # compute weights and contributions for non-exact matches
        diffs = x[:, None] - self.nodes
        weights = jnp.where(exact_matches, jnp.inf, self.weights / diffs)
        contributions = jnp.where(exact_matches, jnp.inf, weights * self.coefficients)

        numerator = jnp.sum(contributions, axis=1)
        denominator = jnp.sum(weights, axis=1)

        result = jnp.where(denominator != 0, numerator / denominator, 0.0)

        # handle exact matches separately
        result = jnp.where(jnp.any(exact_matches, axis=1), self.coefficients[jnp.argmax(exact_matches, axis=1)], result)

        return result
