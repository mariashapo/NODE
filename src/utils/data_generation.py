import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import random, jit, vmap
import numpy as np

from utils.collocation_obj import Collocation
from utils.non_parametric_collocation import collocate_data
#-----------------------------------ODE DEFINITIONS-----------------------------------#
@jit
def harmonic_oscillator(y, t, omega_squared):
    return jnp.array([y[1], -omega_squared * y[0]])

@jit
def damped_oscillation(y, t, damping_factor, omega_squared):
    return jnp.array([y[1], -damping_factor * y[1] - omega_squared * y[0]])

@jit
def van_der_pol(y, t, mu, omega, A = 0):
    """
    Van der Pol oscillator with a periodic forcing term.
    
    Args:
    - y: State vector [y0, y1] where y0 is the displacement and y1 is the velocity.
    - t: Time variable.
    - mu: The damping parameter.
    - A: Amplitude of the forcing term.
    - omega: Angular frequency of the forcing term.
    
    Returns:
    - dydt: Derivatives [dy0/dt, dy1/dt]
    """
    dydt = jnp.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0] + A * jnp.cos(omega * t)])
    return dydt

@jit
def sinusoidal_oscillator(y, t, A, omega):
    return A * jnp.cos(omega * t)

@jit
def cosinusoidal_oscillator(y, t, omega):
    return - omega * jnp.sin(omega * t)

@jit
def decay(y, t, c):
    return -c * y

#---------------------------------------SPACING--------------------------------------#
def generate_chebyshev_nodes(n, start, end):
    # Chebyshev nodes second kind
    k = jnp.arange(n)
    x = jnp.cos(jnp.pi * k / (n - 1))
    nodes = 0.5 * (end - start) * x + 0.5 * (start + end)
    return jnp.sort(nodes)

def legendre_gauss_nodes(n, start, end):
    """
    Compute the Legendre-Gauss nodes for interpolation.
    
    Parameters:
    n (int): Number of nodes.
    start (float): Lower bound of the interval.
    end (float): Upper bound of the interval.
    
    Returns:
    array_like: Legendre-Gauss nodes in the interval [start, end].
    """
    # Use NumPy to find the roots of the Legendre polynomial of degree n
    nodes, _ = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [start, end]
    nodes = 0.5 * (end - start) * (nodes + 1) + start
    return jnp.array(nodes)

#------------------------------------DATA GENERATION---------------------------------#
def generate_ode_data(n_points, noise_level, ode_type, params, start_time=0, end_time=10, spacing_type="equally_spaced", initial_state=None, seed=0, t = None):
    """If *t* is provided, it overrides start_time, end_time, spacing_type, and n_points."""
    
    if initial_state is None:
        if ode_type != "decay":
            initial_state = jnp.array([0.0, 1.0])
        else:
            initial_state = 1.0

    #-----------------------------------------SPACING-------------------------------------#
    if t is None:
        if spacing_type == "equally_spaced" or spacing_type == "uniform":
            t = jnp.linspace(start_time, end_time, n_points, dtype=jnp.float64)
        elif spacing_type == "chebyshev":
            t = generate_chebyshev_nodes(n_points, start_time, end_time)
        else:
            raise ValueError("Unsupported spacing type. Use 'equally_spaced' or 'chebyshev'.")

    #-------------------------------------ODE FUNCTION-------------------------------------#
    if ode_type == "harmonic_oscillator":
        omega_squared = params.get("omega_squared", 1)  # Default omega_squared if not specified
        ode_func = lambda y, t: harmonic_oscillator(y, t, omega_squared)
    elif ode_type == "damped_oscillation":
        damping_factor = params.get("damping_factor", 0.1)  # Default damping factor if not specified
        omega_squared = params.get("omega_squared", 1)  # Default omega_squared if not specified
        ode_func = lambda y, t: damped_oscillation(y, t, damping_factor, omega_squared)
    elif ode_type == "van_der_pol":
        mu = params.get("mu", 1) 
        omega = params.get("omega", 1) 
        ode_func = lambda y, t: van_der_pol(y, t, mu, omega)
    elif ode_type == "decay":
        c = params.get("c", 1) 
        ode_func = lambda y, t: decay(y, t, c)
    elif ode_type == "sinusoidal_oscillator":
        A = params.get("A", 1) 
        omega = params.get("omega", 1) 
        ode_func = lambda y, t: sinusoidal_oscillator(y, t, A, omega)   
    else:
        raise ValueError("Unsupported ODE type provided.")
        
    #-----------------------------------ODEINT SOLUTION----------------------------------#
    y = odeint(ode_func, initial_state, t)

    #-------------------------------------DERIVATIVE-------------------------------------#
    true_derivatives = vmap(lambda y_i, t_i: ode_func(y_i, t_i))(y, t)

    #----------------------------------------NOISE---------------------------------------#
    key = random.PRNGKey(seed)
    y_noisy = y + noise_level * random.normal(key, y.shape)
    
    return t, y, y_noisy, true_derivatives


class DataPreprocessor:
    def __init__(self, data_param):
        self.model_type = 'pyomo'
        self.N = data_param['N']
        self.noise_level = data_param['noise_level']
        self.ode_type = data_param['ode_type']
        self.data_param = data_param['extra_param']
        self.spacing_type = data_param['spacing_type']
        self.start_time = data_param['start_time']
        self.end_time = data_param['end_time']
        self.init_state = data_param['initial_state']
        self.test_size = getattr(data_param, 'test_size', None)
        
    def load_data(self):
        if self.model_type == 'pyomo':
            self.generate_nodes()
        else:
            self.nodes = jnp.linspace(self.start_time, self.end_time, self.N)
        
        # training    
        self.t, self.y, self.y_noisy, true_derivative = generate_ode_data(
            self.N, self.noise_level, self.ode_type, self.data_param, 
            min(self.nodes), max(self.nodes), 
            initial_state = self.init_state, t = self.nodes)
        
        self.true_derivative = true_derivative
        
        if self.test_size is not None:
            test_end_time = self.end_time + self.test_size
        else:
            test_end_time = max(self.nodes) + (max(self.nodes) - min(self.nodes))
        
        # testing
        self.init_state_test = self.y[-1]
        t_test, y_test, _, _ = generate_ode_data(
            self.N*2, self.noise_level, self.ode_type, self.data_param, 
            max(self.nodes), test_end_time, 
            spacing_type = "uniform", 
            initial_state = self.init_state_test)
        
        self.t_test = t_test
        self.y_test = y_test

    def generate_nodes(self):
        collocation = Collocation(self.N, self.start_time, self.end_time, self.spacing_type)
        self.nodes = collocation.compute_nodes()
        self.collocation = collocation
       
    def prepare_collocation(self):
        self.D = np.array(self.collocation.compute_derivative_matrix())
        
    def estimate_derivative(self):
        est_der, est_sol = collocate_data(self.y_noisy, self.t, 'EpanechnikovKernel', bandwidth=0.5)
        self.est_sol = np.array(est_sol)
