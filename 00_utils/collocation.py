import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.linalg import solve
from interpolation import BarycentricInterpolation

jax.config.update("jax_enable_x64", True)

def chebyshev_nodes_second_kind_jax(n, start, stop):
        k = jnp.arange(n, dtype=jnp.float64)  # Ensure 64-bit precision
        x = jnp.cos(jnp.pi * k / (n - 1))
        nodes = 0.5 * (stop - start) * x + 0.5 * (start + stop)
        return jnp.sort(nodes)

def lagrange_basis_node(xi):
    n = len(xi)
    # the basis evaluated at the nodes xi is simply an identity matrix
    L = jnp.eye(n)
    return L

def lagrange_basis(xi, x):
    """"
    Compute the Lagrange basis polynomials at the points x based on the interpolation points xi.
    """
    n = len(xi)
    L = jnp.ones((n, len(x)))
        
    for i in range(n):
        for j in range(n):
            if i != j:
                L = L.at[i, :].set(L[i, :] * (x - xi[j]) / (xi[i] - xi[j]))
    return L

def lagrange_basis_single(xi, x, j):
    """
    Compute the j-th Lagrange basis polynomial at a single point x.

    Parameters:
    xi (array_like): The nodes at which the Lagrange polynomials are defined.
    j (int): Index of the polynomial to compute.
    x (float): The point at which to evaluate the polynomial.

    Returns:
    float: The value of the j-th Lagrange basis polynomial at x.
    """
    n = len(xi)
    L = 1.0
    for m in range(n):
        if m != j:
            L *= (x - xi[m]) / (xi[j] - xi[m])
        # L *= jnp.where(m != j, (x - xi[m]) / (xi[j] - xi[m]), 1.0)
    return L

def derivative_at_node(xi, j, h=1e-5):
    """Approximate the derivative of the j-th Lagrange basis polynomial at its node x_j using central differences."""
    x_j = xi[j]
    forward = lagrange_basis_single(xi, x_j + h, j)
    backward = lagrange_basis_single(xi, x_j - h, j)
    derivative = (forward - backward) / (2 * h)
    return derivative

def compute_weights(xi):
    """
    Compute the weights for each node in the array of nodes xi used in Lagrange interpolation.

    Parameters:
    xi (array_like): The nodes at which the weights are to be computed.

    Returns:
    array_like: An array of weights corresponding to each node.
    """
    xi = jnp.array(xi)
    n = len(xi)
    weights = jnp.zeros(n)

    for j in range(n):
        # exclude the j-th term and compute the product of (x_j - x_m) for all m != j
        terms = xi[j] - jnp.delete(xi, j)
        product = jnp.prod(terms)
        weights = weights.at[j].set(1.0 / product)

    return weights


def lagrange_derivative(xi, weights):
    """
    Compute the derivatives of the Lagrange basis polynomials at the nodes xi.

    Parameters:
    xi (array_like): The nodes at which the Lagrange basis polynomials are defined.
    weights (array_like): The weights for each node in the array of nodes xi.

    Returns:
    array_like: The derivative matrix of the Lagrange basis polynomials at the nodes xi.
    """
    n = len(xi)
    D = jnp.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                D = D.at[i, j].set(weights[j] / weights[i] / (xi[i] - xi[j]))

    # Diagonal elements = the negative sum of the off-diagonal elements in the same row
    for i in range(n):
        D = D.at[i, i].set(-jnp.sum(D[i, :i]) - jnp.sum(D[i, i+1:]))

    return D