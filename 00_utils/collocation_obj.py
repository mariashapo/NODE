import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
# from scipy.optimize import newton
# from numpy.polynomial.legendre import legendre, legder
from scipy.special import roots_legendre, legendre

jax.config.update("jax_enable_x64", True)

class Collocation:
    def __init__(self, n, a = -1, b = 1, spacing_type = "chebyshev", include_zero = True):
        self.n = n
        self.a = a
        self.b = b
        self.spacing_type = spacing_type
        # self.include_zero = include_zero
    
    def compute_nodes(self, unscaled = False):
        if self.spacing_type == "chebyshev":
            # the differential matrix  for chebyshev nodes is computed using the scaled nodes
            # hence, no need for the unscaled nodes
            self.nodes = Collocation.chebyshev_nodes_second_kind(self.n, self.a, self.b)
        elif self.spacing_type == "gauss_legendre":
            if unscaled:
                self.nodes = Collocation.gauss_legendre_nodes(self.n)
            else:
                nodes = Collocation.gauss_legendre_nodes(self.n)
                self.nodes = Collocation.scale_nodes(nodes, self.a, self.b)
        else:
            raise ValueError("Unsupported spacing type. Use 'chebyshev' or 'gauss_legendre'.")
        
        return self.nodes
        
    def compute_derivative_matrix(self, nodes = None):
        if self.spacing_type == "chebyshev":
            # chebyshev nodes are already scaled
            if nodes is not None:
                # nodes can be provided or computed
                self.nodes = nodes
            else:
                self.compute_nodes()
            self.D = Collocation.derivative_matrix_barycentric(self.nodes)
        elif self.spacing_type == "gauss_legendre":
            # compute unscaled and leave the scaling until after the differentiation matrix is computed
            if nodes is not None:
                # these nodes should be unscaled
                self.nodes = nodes
            else:
                self.compute_nodes(unscaled = True)
            self.D = Collocation.scaled_derivative_matrix_gauss_legendre(
                self.nodes, self.a, self.b
                )
            if nodes is None:
                self.compute_nodes()
            else:
                self.nodes = Collocation.scale_nodes(self.nodes, self.a, self.b)
        else:
            raise ValueError("Unsupported spacing type. Use 'chebyshev' or 'gauss_legendre'.")
        
        return self.D
    
    # --------------------------------- Chebyshev Nodes ---------------------------------
    @staticmethod
    def chebyshev_nodes_second_kind(n, start, stop):
        k = jnp.arange(n, dtype=jnp.float64)  
        x = jnp.cos(jnp.pi * k / (n - 1))
        nodes = 0.5 * (stop - start) * x + 0.5 * (start + stop)
        return jnp.sort(nodes)
    
    # --------------------------------- Gauss Legendre Nodes ---------------------------------
    @staticmethod
    def scale_nodes(nodes, a, b):
        # scale the nodes from [-1, 1] to [a, b]
        return 0.5 * (b - a) * (nodes + 1) + a
    
    @staticmethod
    def gauss_legendre_nodes(n):
        
        # _ are the weights
        # these weights are different from the weights for lagrange interpolation
        # these weights are used to compute the integral of a function
        nodes, _ = roots_legendre(n)
        
        return np.sort(nodes)    
    
    # --------------------------------- Barycentric Form --------------------------------- 
    @staticmethod
    def compute_lagrange_weights(nodes):
        """
        Compute the weights for each node in the array of nodes xi used in Lagrange interpolation.

        Parameters:
        nodes (array_like): The nodes at which the weights are to be computed.

        Returns:
        array_like: An array of weights corresponding to each node.
        """
        nodes = jnp.array(nodes)
        n = len(nodes)
        weights = jnp.zeros(n)

        for j in range(n):
            # exclude the j-th term and compute the product of (x_j - x_m) for all m != j
            terms = nodes[j] - jnp.delete(nodes, j)
            product = jnp.prod(terms)
            weights = weights.at[j].set(1.0 / product)

        return weights
        
    @staticmethod
    def derivative_matrix_barycentric(nodes):
        """
        Compute the derivatives of the Lagrange basis polynomials at the nodes xi.

        Parameters:
        nodes (array_like): The nodes at which the Lagrange basis polynomials are defined.
        weights (array_like): The weights for each node in the array of nodes xi.

        Returns:
        array_like: The derivative matrix of the Lagrange basis polynomials at the nodes xi.
        """
        weights = Collocation.compute_lagrange_weights(nodes)
        
        n = len(nodes)
        D = jnp.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    D = D.at[i, j].set(weights[j] / weights[i] / (nodes[i] - nodes[j]))

        # Diagonal elements = the negative sum of the off-diagonal elements in the same row
        for i in range(n):
            D = D.at[i, i].set(-jnp.sum(D[i, :i]) - jnp.sum(D[i, i+1:]))
        
        return D

    # --------------------------------- Gauss Legendre Differential Matrix ---------------------------------  
    
    @staticmethod
    def legendre_poly_deriv(nodes):
        """
        Derivative of the n-th Legendre polynomial at x (Recurrence relations/ Bonnet's recursion formula)
        
        Rerefences:
        # https://math.stackexchange.com/questions/4751256/first-derivative-of-legendre-polynomial
        # https://en.wikipedia.org/wiki/Legendre_polynomials
        """
        n = len(nodes)
        Pn = legendre(n)(nodes)
    
        Pn_minus_1 = legendre(n-1)(nodes) if n > 1 else np.zeros_like(nodes)
        return n / (nodes**2 - 1) * (nodes * Pn - Pn_minus_1)
        
    @staticmethod    
    def derivative_matrix_gauss_legendre(nodes):
        """
        Compute the derivative matrix for Gauss-Legendre nodes.
        
        Reference:
        https://doc.nektar.info/tutorials/latest/fundamentals/differentiation/fundamentals-differentiation.pdf
        """
        Lq_prime_values = Collocation.legendre_poly_deriv(nodes)
        q = len(nodes)
        d_matrix = np.zeros((q, q))
        
        for i in range(q):
            for j in range(q):
                if i != j:
                    d_matrix[i, j] = Lq_prime_values[i] / (Lq_prime_values[j] * (nodes[i] - nodes[j]))
                else:
                    d_matrix[i, j] = nodes[i] / (1 - nodes[i]**2)
        
        return d_matrix
    
    @staticmethod
    def scaled_derivative_matrix_gauss_legendre(nodes, a, b):
        # compute unscaled
        d_matrix = Collocation.derivative_matrix_gauss_legendre(nodes)
        
        # scale differentiation matrix
        scaling_factor = 2 / (b - a)
        scaled_d_matrix = d_matrix * scaling_factor
        
        return scaled_d_matrix
    
    
    
    