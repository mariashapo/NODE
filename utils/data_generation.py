import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import random, jit, vmap
import numpy as np

#-----------------------------------ODE DEFINITIONS-----------------------------------#
@jit
def harmonic_oscillator(y, t, omega_squared):
    return jnp.array([y[1], -omega_squared * y[0]])

@jit
def damped_oscillation(y, t, damping_factor, omega_squared):
    return jnp.array([y[1], -damping_factor * y[1] - omega_squared * y[0]])

@jit
def van_der_pol(y, t, mu):
    return jnp.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])

@jit
def decay(y, t, c):
    return -c * y

#---------------------------------------SPACING--------------------------------------#
def generate_chebyshev_nodes(n, start, end):
    k = jnp.arange(n)
    x = jnp.cos(jnp.pi * k / (n - 1))
    nodes = 0.5 * (end - start) * x + 0.5 * (start + end)
    return jnp.sort(nodes)

#------------------------------------DATA GENERATION---------------------------------#
def generate_ode_data(n_points, noise_level, ode_type, params, start_time=0, end_time=10, spacing_type="equally_spaced", initial_state=None, seed=0):
    
    if initial_state is None:
        if ode_type != "decay":
            initial_state = jnp.array([0.0, 1.0])
        else:
            initial_state = 1.0

    #-----------------------------------------SPACING-------------------------------------#
    if spacing_type == "equally_spaced":
        t = jnp.linspace(start_time, end_time, n_points)
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
        mu = params.get("mu", 1)  # Default mu if not specified
        ode_func = lambda y, t: van_der_pol(y, t, mu)
    elif ode_type == "decay":
        c = params.get("c", 1) 
        ode_func = lambda y, t: decay(y, t, c)   
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