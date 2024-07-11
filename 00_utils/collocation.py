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
    
def chebyshev_nodes_second_kind_jax_32(n, start, stop):
        k = jnp.arange(n, dtype=jnp.float32)  
        x = jnp.cos(jnp.pi * k / (n - 1))
        nodes = 0.5 * (stop - start) * x + 0.5 * (start + stop)
        return jnp.sort(nodes)

def lagrange_basis_node(xi):
    n = len(xi)
    # The basis evaluated at the nodes xi is simply an identity matrix
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
    return D

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


def collocation_ode_solver(ode_system, initial_conditions, t, N, spacing = "Chebyshev", extra_params={}):
    """
    Solves a system of ODEs using polynomial collocation.

    Args:
    - ode_system: Function that takes derivatives, states, parameters and returns the ODE system.
    - initial_conditions: Tuple or list of initial conditions.
    - t: Points at which to evaluate the solution.
    - N: Number of collocation points.
    - extra_params: Dictionary of extra parameters required by the ODE system.

    Returns:
    - solution: Array of shape (len(t), 2)
    """
    T_start, T_end = t[0], t[-1]
    
    if spacing == "Chebyshev":
         collocation_points = BarycentricInterpolation(N, kind='chebyshev2', start=T_start, stop=T_end).nodes
         # chebyshev_nodes_second_kind(N, T_start, T_end)
    if spacing == "Chebyshev_local":
         collocation_points = chebyshev_nodes_second_kind_jax(N, T_start, T_end)
    if spacing == "Chebyshev_local_32":
         collocation_points = chebyshev_nodes_second_kind_jax_32(N, T_start, T_end)
    if spacing == "Linear":
        collocation_points = jnp.linspace(T_start, T_end, N)
        
    # Evaluate basis functions and their derivatives at collocation points
    phi = jnp.eye(N)
    weights = compute_weights(collocation_points)
    dphi_dt = lagrange_derivative(collocation_points, weights)

    # Form the collocation matrix based on the given ODE system
    A = ode_system(dphi_dt, phi, extra_params)

    b = jnp.zeros(2 * (N))
    b_aug = jnp.concatenate([b, jnp.array(initial_conditions)])

    # Incorporate initial conditions
    I_x1 = jnp.zeros((1, 2 * (N)))
    I_x1 = I_x1.at[0, :N].set(phi[0, :])

    I_x2 = jnp.zeros((1, 2 * (N)))
    I_x2 = I_x2.at[0, N:].set(phi[0, :])

    A_aug = jnp.vstack([A, I_x1, I_x2])

    # Solve the system using the normal equations for stability
    c = solve(A_aug.T @ A_aug, A_aug.T @ b_aug)

    c1 = c[:N]
    c2 = c[N:]
    
    lb = jnp.transpose(lagrange_basis(collocation_points, t))
    x1 = lb @ c1
    x2 = lb @ c2

    solution = jnp.vstack([x1, x2]).T
    return solution


if __name__ == "__main__":
    #---------------------------------PROBLEM DEF---------------------------------------------------------#
    # Example ODE system
    def example_ode_system(dphi_dt, phi, params):
        omega = params.get('omega', 1.0)
        return jnp.block([
            [dphi_dt, -phi],
            [omega**2 * phi, dphi_dt]
        ])

    t_span = (0, 10)
    N = 20
    omega = 2.0
    initial_conditions = (1.0, 0.0)  # x1_0, x2_0
    params = {'omega': omega}
    
    x1, x2 = collocation_ode_solver(example_ode_system, initial_conditions, t_span, N, params)
    print(x1, x2)