import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
from solvers import rk4_solver, rk4_step
import matplotlib.pyplot as plt
from interpolation import BarycentricInterpolation
from jax.scipy.linalg import solve
import time

def lagrange_basis_node(xi):
    n = len(xi)
    # The basis evaluated at the nodes xi is simply an identity matrix
    L = jnp.eye(n)
    return L

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
    return D

def construct_matrix_and_vector(xi, y0, LHS, RHS):
    """
    Construct the matrix A and vector b for solving an ODE using Lagrange basis polynomials,
    incorporating an initial value condition.

    Parameters:
    xi (array_like): Nodes at which the values of the ODE solution are known or to be calculated.
    y0 (float): The initial value at the first node xi[0].

    Returns:
    tuple: A tuple containing the matrix A and vector b for the ODE problem.
    """
    n = len(xi)
    b = jnp.zeros(n)
    
    # basis is just the identity matrix!
    # basis = jnp.eye(n)
    weights = compute_weights(xi)
    derivatives = lagrange_derivative(xi, weights)
    
    # SPECIFIC TO THE ODE; 
    # MUST BE MADE GENERAL FOR ANY ODE
    A = LHS(xi, derivatives)
    
    # initial condition
    A = A.at[0, :].set(0)  
    A = A.at[0, 0].set(1) 
    # Set the initial condition 
    # 1y0 + 0y1 + 0y2 + ... = y0

    b = RHS(xi) # genrate vector b
    b = b.at[0].set(y0) # init condition
    # b[0] = y0
    
    return A, b

def collocation_solver(start, stop, y0, n, LHS, RHS, x_eval):
    """
    Solve the ODE using collocation method with n nodes.
    
    Inputs:
    start (float): The start of the interval.
    stop (float): The end of the interval.
    y0 (float): The initial value at the start.
    n (int): The number of collocation nodes.
    x_eval (array_like): The points at which to evaluate the solution.
    """  
    interpolator = BarycentricInterpolation(n=n, kind='chebyshev2', start=start, stop=stop)
    xi = interpolator.nodes

    A, b = construct_matrix_and_vector(xi, y0, LHS, RHS)
    y_coeff = find_solution(A, b)
    
    interpolator.fit(y_coeff)
    result = interpolator.evaluate(x_eval)
    return result
    
def find_solution(A, b):
    """
    Solve the linear system of equations A * y = b for y.
    """
    y = solve(A, b)
    return y 
    
if __name__ == "__main__":
    #---------------------------------PROBLEM DEF---------------------------------------------------------#
    def LHS(xi, derivatives):
        "LHS of the ODE"
        basis = jnp.eye(len(xi)) 
        return derivatives + 3 * basis
    
    def RHS(xi):
        "RHS of the ODE"
        return 2 * jnp.exp(-xi) * jnp.sin(xi) + jnp.exp(-xi) * jnp.cos(xi)
    
    # rearrange the ODE to the form y' = f(y, t) to be used in the ODE solver
    def ODE(y, t):
        "ODE of the form y' = f(y, t)"
        return 2 * jnp.exp(-t) * jnp.sin(t) + jnp.exp(-t) * jnp.cos(t) - 3 * y
    
    # in case the exact solution is known (used for the plot)
    def solution(t):
        return jnp.exp(-t) * jnp.sin(t)
    
    #---------------------------------PROBLEM INIT---------------------------------------------------------#
    start = 0
    stop = 10
    y0 = 0
    n = 24 # number of collocation nodes
    steps = 1000
    t = np.linspace(start, stop, steps)
    
    start_time = time.time()
    collocation_values = collocation_solver(start, stop, y0, n, LHS, RHS, t)
    end_time = time.time()
    print(f"Elapsed time for collocation: {end_time - start_time} seconds")
    
    start_time = time.time()
    times, rk4_values = rk4_solver(ODE, y0, start, stop, steps)
    end_time = time.time()
    print(f"Elapsed time for RK4 solver: {end_time - start_time} seconds")
    
    y0 = jnp.array([0.0])
    start_time = time.time()
    odeint_values = odeint(ODE, y0, t)
    end_time = time.time()
    print(f"Elapsed time for Odeint() solver: {end_time - start_time} seconds")

    true_values = solution(t)
    
    def calculate_mae(predictions, true_values):
        predictions = jnp.array(predictions).squeeze()
        true_values = jnp.array(true_values).squeeze()
        return jnp.mean(jnp.abs(predictions - true_values))
    
    rmse_collocation = calculate_mae(collocation_values, true_values)
    print(f"MAE for Collocation: {rmse_collocation}")
    
    rmse_rk4 = calculate_mae(rk4_values, true_values)
    print(f"MAE for RK4: {rmse_rk4}")
    
    rmse_odeint = calculate_mae(odeint_values, true_values)
    print(f"MAE for Odeint: {rmse_odeint}")
        
    t_new = np.linspace(start, stop, 1000)
    true_values = solution(t_new)
    plt.figure(figsize=(10, 7))
    
    plt.plot(t_new, true_values, label='Exact Solution')
    plt.plot(t, collocation_values, '--', label='Collocation Solution')
    plt.plot(t, rk4_values, '--', label='RK4 Solution')
    plt.plot(t, odeint_values, '--', color = 'red', label='Odeint Solution')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Collocation Solution vs Exact Solution')
    plt.legend()
    plt.grid(True)
    plt.show()
    
