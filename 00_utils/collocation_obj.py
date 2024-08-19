import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
# from scipy.optimize import newton
# from numpy.polynomial.legendre import legendre, legder
from scipy.special import roots_legendre, legendre, eval_legendre
from scipy.special import roots_jacobi

jax.config.update("jax_enable_x64", True)

class Collocation:
    def __init__(self, n, a = -1, b = 1, spacing_type = "chebyshev", include_init = False):
        """
        Args:
            spacing_type (str, optional): _description_. Defaults to "chebyshev".
            ["chebyshev", "gauss_legendre", "gauss_radau", "gauss_lobatto"]

        """
        self.n = n
        self.a = a
        self.b = b
        self.spacing_type = spacing_type
        self.include_init = include_init
    
    def compute_nodes(self, unscaled = False):
        if self.spacing_type == "chebyshev":
            # the differential matrix  for chebyshev nodes is computed using the scaled nodes
            # hence, no need for the unscaled nodes
            self.nodes = Collocation.chebyshev_nodes_second_kind(self.n, self.a, self.b)
            
        elif self.spacing_type in ['gauss_legendre', 'gauss_radau', 'gauss_lobatto']:
            self.nodes = self.legendre_nodes(self.n, self.a, self.b, self.spacing_type)
        
        # old method for gauss_legendre nodes   
        elif self.spacing_type == "gauss_legendre_old":
            if unscaled:
                self.nodes = self.gauss_legendre_nodes(self.n)
            else:
                nodes = self.gauss_legendre_nodes(self.n)
                self.nodes = Collocation.scale_nodes(nodes, self.a, self.b)
                
        else:
            raise ValueError("Unsupported spacing type. Use 'chebyshev', 'gauss_legendre', 'gauss_radau', 'gauss_lobatto'.")
        
        return self.nodes
        
    def compute_derivative_matrix(self, nodes = None):
        if self.spacing_type in ['chebyshev', 'gauss_legendre', 'gauss_radau', 'gauss_lobatto']:
            # chebyshev nodes are already scaled
            if nodes is not None:
                # nodes can be provided or computed
                self.nodes = nodes
            else:
                self.compute_nodes()
            self.D = Collocation.derivative_matrix_barycentric_simplfied(self.nodes)
            
        elif self.spacing_type == "gauss_legendre_old":
            # compute unscaled and leave the scaling until after the differentiation matrix is computed
            if nodes is not None:
                # these nodes should be unscaled
                self.nodes = nodes
            else:
                self.compute_nodes(unscaled = True)
            self.D = self.scaled_derivative_matrix_gauss_legendre(
                self.nodes, self.a, self.b
                )
            if nodes is None:
                self.compute_nodes()
            else:
                self.nodes = Collocation.scale_nodes(self.nodes, self.a, self.b)
        else:
            raise ValueError("Unsupported spacing type. Use 'chebyshev', 'gauss_legendre', 'gauss_radau', 'gauss_lobatto'.")
        
        return self.D
    
    # -------------------------------------------- Chebyshev Nodes --------------------------------------------
    @staticmethod
    def chebyshev_nodes_second_kind(n, start, stop):
        k = jnp.arange(n, dtype=jnp.float64)  
        nodes = jnp.cos(jnp.pi * k / (n - 1))
        
        nodes = Collocation.scale_nodes(nodes, start, stop)
        
        return jnp.sort(nodes)
    
    # ---------------- gauss_legendre (a,b), gauss_radau [a,b),  gauss_lobatto [a,b]----------------------
    @staticmethod
    def legendre_nodes(n, start, stop, spacing_type):
        # https://scicomp.stackexchange.com/questions/32918/need-an-example-legendre-gauss-radau-pseudospectral-differentiation-matrix-or-ma
        
        if spacing_type == "gauss_legendre":
            method = 1
        elif spacing_type == "gauss_lobatto":
            method = 2
        elif spacing_type == "gauss_radau":
            method = 3
        
        # Define boundary conditions, 1 for non-included and 0 for included boundary
        na = [1, 0, 0] 
        nb = [1, 0, 1] 
        
        alpha = 1.0 - na[method-1]
        beta = 1.0 - nb[method-1]
        
        # Adjust the number of nodes to account for the boundary conditions
        n -= (alpha + beta)

        # get roots and weights for the Jacobi polynomial
        roots, weights = roots_jacobi(n, alpha, beta)

        nodes = Collocation.scale_nodes(roots, start, stop)
        
        if alpha != 0:
            nodes = jnp.hstack(([start], nodes)) 
        if beta != 0:
            nodes = jnp.hstack((nodes, [stop]))
        
        return jnp.sort(nodes)
    
    # --------------------------------- Gauss Legendre-Gaus Nodes (Old Method) ---------------------------------
    @staticmethod
    def scale_nodes(nodes, a, b):
        # scale the nodes from [-1, 1] to [a, b]
        return 0.5 * (b - a) * (nodes + 1) + a
    
    # @staticmethod
    def gauss_legendre_nodes(self, n):
        
        # _ are the weights
        # these weights are different from the weights for lagrange interpolation
        # these weights are used to compute the integral of a function
        
        if self.include_init:
            nodes, _ = roots_legendre(n - 1)
            nodes = np.insert(nodes, 0, -1)
        else:
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
    
    @staticmethod
    def derivative_matrix_barycentric_simplfied(nodes):
        # simpler version of the derivative matrix calculation based on:
        # https://scicomp.stackexchange.com/questions/32918/need-an-example-legendre-gauss-radau-pseudospectral-differentiation-matrix-or-ma
        
        node_differences = nodes[:, jnp.newaxis] - nodes
        # prevent division by zero 
        node_differences = node_differences.at[jnp.diag_indices_from(node_differences)].set(1)
        product_of_differences = jnp.prod(node_differences, axis=1)
        derivative_matrix = product_of_differences[:, jnp.newaxis] / (product_of_differences * node_differences)
        diagonal_adjustments = 1 - jnp.sum(derivative_matrix, axis=1)
        derivative_matrix = derivative_matrix.at[jnp.diag_indices_from(derivative_matrix)].set(diagonal_adjustments)
        
        return derivative_matrix

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
        # nodes**1 -1 becomes zero if nodes = -1 or nodes = 1
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
    def legendre_polynomial(n, x):
        if n == 0:
            return 1
        elif n == 1:
            return x

        Pnm1 = x # P_0(x)
        Pnm2 = 1 # P_1(x)
        Pn = None

        for k in range(2, n+1):
            Pn = ((2*k - 1) * x * Pnm1 - (k - 1) * Pnm2) / k
            Pnm2 = Pnm1
            Pnm1 = Pn

        return Pn

    @staticmethod
    def derivative_matrix_gauss_radau_legendre(nodes):
        N = len(nodes)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    LNi = Collocation.legendre_polynomial(N - 1, nodes[i])
                    LNj = Collocation.legendre_polynomial(N - 1, nodes[j])

                    D[i, j] = (LNi / LNj) * ((1 - nodes[j])/(1 - nodes[i])) * (1 / (nodes[i] - nodes[j]))

        for i in range(N):
            if i == 0:
                D[i, i] = - ((N + 1)* (N - 1)) / 4
            else:
                D[i, i] = 1 / (2 * (1 - nodes[i])) 

        return D

    @staticmethod
    def derivative_matrix_gauss_lobatto_legendre(nodes):
        N = len(nodes)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    LNi = Collocation.legendre_polynomial(N - 1, nodes[i])
                    LNj = Collocation.legendre_polynomial(N - 1, nodes[j])

                    D[i, j] = (LNi / LNj) * (1 / (nodes[i] - nodes[j]))

        # Calculate diagonal elements based on the off-diagonal computed values
        for i in range(N):
            if i == 0:
                D[i, i] = - (N * (N - 1)) / 4
            elif i == N - 1:
                D[i, i] = (N * (N - 1)) / 4
            else:
                D[i, i] = -np.sum(D[i, :])

        return D
    
    def scaled_derivative_matrix_gauss_legendre(self, nodes, a, b):
        # compute unscaled
        d_matrix = Collocation.derivative_matrix_gauss_legendre(nodes)
        
        # scale differentiation matrix
        scaling_factor = 2 / (b - a)
        scaled_d_matrix = d_matrix * scaling_factor
        
        return scaled_d_matrix
    
    
    
    