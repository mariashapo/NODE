import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.experimental.ode import odeint

#-----------------------------------COLLOCATION FUNCTIONS--------------------------------#

def construct_t_matrix(order, tpoints, t):
    """
    Constructs the T matrix for collocation.

    Args:
        order (int): The order of the T matrix (1 or 2).
        tpoints (jax.numpy.ndarray): The array of time points.
        t (float): The current time point.

    Returns:
        jax.numpy.ndarray: The constructed T matrix.
    """
    if order == 1:
        return jnp.vstack([jnp.ones_like(tpoints), tpoints - t]).T
    elif order == 2:
        return jnp.vstack([jnp.ones_like(tpoints), tpoints - t, (tpoints - t)**2]).T

def construct_w(tpoints, t, bandwidth, kernel_type):
    """
    Constructs the weight matrix W for collocation.

    Args:
        tpoints (jax.numpy.ndarray): The array of time points.
        t (float): The current time point.
        bandwidth (float): The bandwidth for the kernel.
        kernel_type (str): The type of kernel to use ("TriangularKernel" or "EpanechnikovKernel").

    Returns:
        jax.numpy.ndarray: The diagonal weight matrix W.
    """
    return jnp.diag(jax.vmap(lambda tp: calckernel(kernel_type, (tp - t) / bandwidth))(tpoints))

def calckernel(kernel_type, x):
    """
    Calculates the kernel value.

    Args:
        kernel_type (str): The type of kernel to use ("TriangularKernel" or "EpanechnikovKernel").
        x (float): The input value for the kernel function.

    Returns:
        float: The calculated kernel value.
    """
    if kernel_type == "TriangularKernel":
        return jnp.maximum((1 - jnp.abs(x)), 0)
    elif kernel_type == "EpanechnikovKernel":
        return (3/4) * jnp.maximum((1 - x**2), 0)
    
def configure_extraction_vectors(data):
    """
    Configures the extraction vectors e1 and e2.

    Args:
        data (jax.numpy.ndarray): The data array to infer the shape.

    Returns:
        tuple: The extraction vectors e1 and e2.
    """
    _one = 1.0
    _zero = 0.0

    # To extract the constant term (intercept)
    e1 = jnp.array([[_one, _zero]]).T  
    # To extract the linear term and optionally the quadratic term
    e2 = jnp.array([[_zero, _one, _zero]]).T 
    
    return e1, e2

def collocate_data(data, tpoints, kernel="TriangularKernel", bandwidth=None):
    """Applies the kernel-based collocation method to smooth data.

    Args:
        data (jax.numpy.ndarray): The noisy data array.
        tpoints (jax.numpy.ndarray): The array of time points.
        kernel (str, optional): The type of kernel to use ("EpanechnikovKernel" or "TriangularKernel").
        bandwidth (float, optional): The bandwidth for the kernel. If None, it is calculated automatically. Defaults to None.

    Returns:
        tuple: The estimated derivatives and smoothed solutions.
    """
    n = len(tpoints)

    # the exprected shape is (dimensions, no of datapoints)
    # hence, the number of datapoints should be larger than the number of dimensions
    if len(data.shape)>1 and data.shape[0] > data.shape[1]:
        data = data.T
#-----------------------------------BANDWIDTH----------------------------------#
    if bandwidth is None:
        bandwidth = (n**(-1/5)) * (n**(-3/35)) * (jnp.log(n)**(-1/16))
    
    # M = construct_t_matrix(2, tpoints, 0)

#-------------------------COEFFICIENT EXTRACTION VECTORS------------------------#
    # later on used to extract  coefficients from local polynomial approximations
    e1, e2 = configure_extraction_vectors(data[:, 0])

    # print(e2.shape)
    estimated_solution = []
    estimated_derivative = []

    for _t in tpoints:
#-----------------------------MATRIX CONSTRUCTION--------------------------------#
        # constructs the first-order T matrix
        T1 = construct_t_matrix(1, tpoints, _t)
        # constructs the second-order T matrix
        T2 = construct_t_matrix(2, tpoints, _t)
        # constructs the weight matrix
        W = construct_w(tpoints, _t, bandwidth, kernel)
        # print(W)
#-----------------------------------WEIGHT DATA-----------------------------------#
        # print(W.shape)
        # print(data.T.shape)
        Wd = W @ data.T
        # return 0, 0

#---------------------------------WEIGHT T-MATRICES-------------------------------#
        WT1 = W @ T1
        WT2 = W @ T2
        
        # form matrices for the least-squares weighted quadratic fitting problem
        T1WT1 = T1.T @ WT1
        T2WT2 = T2.T @ WT2
#---------------------------------------------------------------------------------#
#                           Solve (T1^T W T1 c1) = T1^T W y 
#---------------------------------------------------------------------------------#

#------------------------------EXTRACT COEFFICIENTS------------------------------#

        solution1 = solve(T1WT1, T1.T @ Wd)
        result_solution = e1.T @ solution1
        result_solution = jnp.squeeze(result_solution) # remove extra dimension
        estimated_solution.append(result_solution)
        
        solution2 = solve(T2WT2, T2.T @ Wd)
        result_derivative = (e2.T @ solution2)
        result_derivative = jnp.squeeze(result_derivative) # remove extra dimension
        estimated_derivative.append(result_derivative)
        # print(result_derivative)
        
        estimated_solution_ = jnp.stack(estimated_solution).T
        estimated_derivative_ = jnp.stack(estimated_derivative).T

    return estimated_derivative_, estimated_solution_
    