import numpy as np

class BarycentricInterpolation:
    def __init__(self, n, kind='chebyshev2', start=-5, stop=5):
        self.n = n
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nodes = self.chebyshev_nodes_second_kind()
        self.weights = self.compute_barycentric_weights()
        self.coefficients = None

    def chebyshev_nodes_second_kind(self):
        k = np.arange(self.n)
        x = np.cos(np.pi * k / (self.n - 1))
        nodes = 0.5 * (self.stop - self.start) * x + 0.5 * (self.start + self.stop)
        return np.sort(nodes)

    def compute_barycentric_weights(self):
        w = np.ones(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    w[i] /= (self.nodes[i] - self.nodes[j])
        return w

    def fit(self, y):
        if len(y) != self.n:
            raise ValueError("The length of y must match the number of nodes.")
        self.coefficients = np.array(y)

    def evaluate(self, x):
        x = np.asarray(x)
        numerator = np.zeros_like(x, dtype=float)
        denominator = np.zeros_like(x, dtype=float)
        exact_matches = np.isclose(x[:, None], self.nodes)

        # Compute weights and contributions for non-exact matches
        diffs = x[:, None] - self.nodes
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = self.weights / diffs
            weights[exact_matches] = np.inf  # Assign infinite weight for exact matches
            contributions = weights * self.coefficients

        numerator = np.sum(contributions, axis=1)
        denominator = np.sum(weights, axis=1)

        result = numerator / denominator
        for i in range(len(x)):
            if np.any(exact_matches[i]):
                result[i] = self.coefficients[np.argmax(exact_matches[i])]
        return result