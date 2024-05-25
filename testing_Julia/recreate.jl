using DifferentialEquations
using Flux: Chain, Dense, tanh, params
using Flux
using Optim
using DiffEqFlux
using DiffEqParamEstim
using Plots

# Define initial conditions and parameters
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)

# Define the true ODE function
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

# Generate time points and solve the ODE
t = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

# Define the neural ODE using Flux.Dense
dudt = Chain(x -> x.^3,
             Dense(2, 50, tanh),
             Dense(50, 2))
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=t)

# Get parameters of the neural network
ps = Flux.params(dudt)

# Define the prediction function
function predict_n_ode(p)
    n_ode(u0, p)
end

# Custom two-stage function
function node_two_stage_function(model, x, tspan, saveat, ode_data, args...; kwargs...)
    dudt_(du, u, p, t) = du .= model(u)
    prob_fly = ODEProblem(dudt_, x, tspan)
    two_stage_method(prob_fly, saveat, ode_data)
end

# Dummy definition of two_stage_method (for the script to be self-contained)
function two_stage_method(prob, saveat, ode_data)
    sol = solve(prob, Tsit5(), saveat=saveat)
    predicted_data = Array(sol)
    error = ode_data .- predicted_data
    cost = sum(abs2, error)
    return CollocationResult(() -> cost)
end

# Dummy definition of CollocationResult (for the script to be self-contained)
struct CollocationResult
    cost_function::Function
end

node_two_stage = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
loss_n_ode = node_two_stage.cost_function

# Define the loss function using the two-stage method
function los(p)
    l = loss_n_ode(p)
    pred = predict_n_ode(p)
    return l, pred
end

# Define the callback function
cb = function (p, l, pred)
    println("Loss: ", l)
    pl = scatter(t, ode_data[1, :], label="Data (u1)")
    scatter!(pl, t, pred[1, :], label="Prediction (u1)")
    plot!(pl, t, ode_data[2, :], label="Data (u2)")
    scatter!(pl, t, pred[2, :], label="Prediction (u2)")
    display(pl)
end

# Display initial ODE with initial parameters
cb(ps, los(ps)...)

# Train the neural ODE
res1 = DiffEqFlux.sciml_train!(los, ps, ADAM(0.05), cb = cb, maxiters = 100)
