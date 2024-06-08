from collocation import *
from solvers import *

def calculate_mae(predictions, true_values):
    predictions = jnp.array(predictions).squeeze()
    true_values = jnp.array(true_values).squeeze()
    return jnp.mean(jnp.abs(predictions - true_values))

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

#---------------------------------PROBLEM DEF-----

def LHS(xi, derivatives):
    "LHS of the ODE representing exponential growth"
    basis = jnp.eye(len(xi))  # This is usually identity matrix for simplicity
    return derivatives  # This function returns the derivatives as is, no added term

def RHS(xi):
    "RHS of the exponential growth ODE"
    k = 3  # Growth rate
    return k * xi  # The function form for exponential growth: k * y

# Rearrange the ODE to the form y' = f(y, t) to be used in an ODE solver
def ODE(y, t):
    "ODE of the form y' = ky"
    k = 3  # Growth rate
    return k * y  # The model of exponential growth

# In case the exact solution is known (used for the plot and verification)
def solution(t, y0=1):
    k = 3  # Growth rate
    return y0 * jnp.exp(k * t)  # The solution to y' = ky with initial condition y0 at t=0

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

#---------TRUE VALUES-----------#
true_values = solution(t)
#-------------------------------#

#--------------------DIFFERENT COLLOCATION NODES-----------------------#
"""node_range = range(4, 25)  # From 4 to 24 nodes
mape_results = []
time_results = []

for n in node_range:
    start_time = time.time()
    collocation_values = collocation_solver(start, stop, y0, n, LHS, RHS, t)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    mape = calculate_mae(collocation_values, true_values)

    time_results.append(elapsed_time)
    mape_results.append(mape)

def plot():
    # Plotting MAPE
    plt.figure(figsize=(10, 6))
    plt.plot(list(node_range), mape_results, linestyle='-', color='b')
    plt.title('MAE vs. Number of Collocation Nodes')
    plt.xlabel('Number of Collocation Nodes')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.show()

    # Plotting Computation Time
    plt.figure(figsize=(10, 6))
    plt.plot(list(node_range), time_results, linestyle='-', color='r')
    plt.title('Computation Time vs. Number of Collocation Nodes')
    plt.xlabel('Number of Collocation Nodes')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.show()

plot()"""

#----------------------------RK4 and Odeint------------------------------------------#

start_time = time.time()
times, rk4_values = rk4_solver(ODE, y0, start, stop, steps)
end_time = time.time()
print(f"Elapsed time for RK4 solver: {end_time - start_time} seconds")

y0 = jnp.array([0.0])
start_time = time.time()
odeint_values = odeint(ODE, y0, t)
end_time = time.time()
print(f"Elapsed time for Odeint() solver: {end_time - start_time} seconds")

#----------------------------DIFFERENT NUMBER OF STEPS------------------------------------------#
"""start = 0
stop = 10
steps_options = [100, 200, 500, 1000, 2000, 5000]  # Different step sizes to test
true_values = solution(np.linspace(start, stop, 5000))  # High-resolution solution for reference

#-----------------------------------------------------------------------------------------------#
rk4_times = []
rk4_maes = []
odeint_times = []
odeint_maes = []

for steps in steps_options:
    t = np.linspace(start, stop, steps)
    y0 = 0
    
    # RK4 Solver
    start_time = time.time()
    _, rk4_values = rk4_solver(ODE, y0, start, stop, steps)
    rk4_end_time = time.time()
    
    # Odeint Solver
    y0 = jnp.array([0.0])
    odeint_start_time = time.time()
    odeint_values = odeint(ODE, y0, t)
    odeint_end_time = time.time()
    
    # Calculate MAE
    true_interpolated = np.interp(t, np.linspace(start, stop, 5000), true_values)  # Interpolate true values to current t
    
    rk4_mae = calculate_mae(rk4_values, true_interpolated)
    odeint_mae = calculate_mae(odeint_values, true_interpolated)
    
    # Store results
    rk4_times.append(rk4_end_time - start_time)
    rk4_maes.append(rk4_mae)
    odeint_times.append(odeint_end_time - odeint_start_time)
    odeint_maes.append(odeint_mae)

# Plotting MAE
plt.figure(figsize=(10, 6))
plt.plot(steps_options, rk4_maes, marker='o', linestyle='-', color='b', label='RK4 MAE')
plt.plot(steps_options, odeint_maes, marker='x', linestyle='-', color='r', label='Odeint MAE')
plt.title('MAE vs. Steps for RK4 and Odeint')
plt.xlabel('Number of Steps')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Computation Time
plt.figure(figsize=(10, 6))
plt.plot(steps_options, rk4_times, marker='o', linestyle='-', color='b', label='RK4 Computation Time')
plt.plot(steps_options, odeint_times, marker='x', linestyle='-', color='r', label='Odeint Computation Time')
plt.title('Computation Time vs. Steps for RK4 and Odeint')
plt.xlabel('Number of Steps')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()"""
#-----------------------------------------------------------------------------------------------#

rmse_collocation = calculate_mae(collocation_values, true_values)
print(f"MAE for Collocation: {rmse_collocation}")

rmse_rk4 = calculate_mae(rk4_values, true_values)
print(f"MAE for RK4: {rmse_rk4}")

rmse_odeint = calculate_mae(odeint_values, true_values)
print(f"MAE for Odeint: {rmse_odeint}")
    
plt.figure(figsize=(15, 8))

plt.plot(t, true_values, label='Exact Solution')
plt.plot(t, collocation_values, '--', label='Collocation Solution')
plt.plot(t, rk4_values, '--', label='RK4 Solution')
plt.plot(t, odeint_values, '--', color = 'red', label='Odeint Solution')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Collocation Solution vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()