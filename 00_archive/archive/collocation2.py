import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from solvers import rk4_solver, rk4_step
import matplotlib.pyplot as plt
from interpolation import BarycentricInterpolation
from jax.scipy.linalg import solve
import time

# @jit
def lagrange_basis(xi, x):
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

# approximate the derivative of the j-th Lagrange basis polynomial at its node x_j using central differences
def derivative_at_node(xi, j, h=1e-5):
    """Approximate the derivative of the j-th Lagrange basis polynomial at its node x_j using central differences."""
    x_j = xi[j]
    forward = lagrange_basis_single(xi, x_j + h, j)
    backward = lagrange_basis_single(xi, x_j - h, j)
    derivative = (forward - backward) / (2 * h)
    return derivative

# @jit
def compute_weights(xi):
    """
    Compute the weights for each node in the array of nodes xi used in Lagrange interpolation.

    Parameters:
    xi (array_like): The nodes at which the weights are to be computed.

    Returns:
    array_like: An array of weights corresponding to each node.
    """
    n = len(xi)
    xi = jnp.array(xi) 
    weights = jnp.zeros(n)

    for j in range(n):
        # Exclude the j-th term and compute the product of (x_j - x_m) for all m != j
        terms = xi[j] - jnp.delete(xi, j)
        product = jnp.prod(terms)
        weights = weights.at[j].set(1.0 / product)

    return weights

# @jit
def lagrange_derivative(xi, weights):
    n = len(xi)
    D = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D = D.at[i, j].set(weights[j] / weights[i] / (xi[i] - xi[j]))
            else:
                # approximation to handle the diagonal case; it needs proper handling for exact values!!!
                # D = D.at[i, j].set(0)
                approx_derivative = derivative_at_node(xi, j)
                D = D.at[i, j].set(approx_derivative)