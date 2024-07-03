import numpy as np
import jax.numpy as jnp

def rk4_solver(f, y0, start, stop, steps, *args, **kwargs):
    """
    Solve an ODE with the RK4 method from time start to stop with initial condition y0.
    """
    t = np.linspace(start, stop, steps)
    dt = t[1] - t[0]
    times = [start]
    values = []
    values.append(y0)
    t = start
    y = y0
    for _ in range(1, steps):
        y = rk4_step(f, y, t, dt, *args, **kwargs)
        t += dt
        times.append(t)
        # print(type(y))
        values.append(y)

    return times, values


def rk4_step(f, y, t, dt, *args, **kwargs):
    """
    Perform a single step of the RK4 integration.

    Parameters:
    - f: The derivative function of the system, should accept at least two parameters (y, t) and additional parameters
    - y: Current value of the dependent variable
    - t: Current time
    - dt: Step size
    - args, kwargs: Additional arguments and keyword arguments for the derivative function f

    Returns:
    - y_next: Next value of y after a step dt
    """
    k1 = dt * f(y, t, *args, **kwargs)
    k2 = dt * f(y + 0.5 * k1, t + 0.5 * dt, *args, **kwargs)
    k3 = dt * f(y + 0.5 * k2, t + 0.5 * dt, *args, **kwargs)
    k4 = dt * f(y + k3, t + dt, *args, **kwargs)
    y_next = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    if isinstance(y_next, (int, float, complex)):
        return y_next
    else:
        return y_next.item()

if __name__ == "__main__":
    def simple_ode(y, t):
        return -2 * y
    
    y0 = 1.0  # Example initial condition
    start = 0.0  # Start of the interval
    stop = 2.0  # End of the interval
    steps = 10  # Number of steps

    # Call the RK4 solver
    times, values = rk4_solver(simple_ode, y0, start, stop, steps)

    print(len(times))
    print(values)
    # Print out the times and values
    for t, y in zip(times, values):
        print(f"time: {t:.2f}, y: {y:.4f}")