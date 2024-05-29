import numpy as np

class BarycentricInterpolation:
    def __init__(self, n, kind='chebyshev2', start=-5, stop=5):
        self.n = n
        self.kind = kind
        self.start = start
        self.stop = stop
        self.nodes = self.chebyshev_nodes_second_kind()
        self.weights = self.compute_barycentric_weights()

    def chebyshev_nodes_second_kind(self):
        k = np.arange(self.n) # 0, 1, 2, ..., n-1
        x = np.cos(np.pi * k / (self.n - 1)) # Chebyshev nodes in [-1, 1]
        nodes = 0.5 * (self.stop - self.start) * x + 0.5 * (self.start + self.stop) # Scale and shift to [start, stop]
        return np.sort(nodes) # ensure nodes are sorted

    def compute_barycentric_weights(self):
        n = len(self.nodes)
        # for each node
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] /= (self.nodes[i] - self.nodes[j])
        return w

    def interpolate(self, y, x):
        """Interpolate function values `y` at nodes `self.nodes` to points `x`."""
        numer = 0.0
        denom = 0.0
        close_node_index = None
        close_node_diff = float('inf')
        
        # Check if x is close to any node and handle that case by direct assignment
        for j in range(self.n):
            diff = x - self.nodes[j]
            if abs(diff) < 1e-10:  # small threshold to detect closeness
                close_node_index = j
                close_node_diff = diff
                break  # break if exact node match is found to avoid divide by zero
            weights = self.weights[j] / diff
            numer += weights * y[j]
            denom += weights

        if close_node_index is not None:
            if close_node_diff == 0:
                return y[close_node_index]  # return the exact value at the node
            else:
                return numer / denom  # proceed with normal computation if not perfectly close
        return numer / denom if denom != 0 else 0.0  # prevent division by zero