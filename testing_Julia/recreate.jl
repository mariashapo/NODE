using DifferentialEquations
using Lux, DiffEqFlux, OrdinaryDiffEq, DiffEqSensitivity
using Flux: Chain, Dense, tanh
using Optimization, OptimizationOptimisers
using Plots

# Define initial conditions and parameters
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
t = range(tspan[1], tspan[2], length=datasize)

# Define the true ODE function
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

# Generate time points and solve the ODE
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), saveat=t))

# Define the neural ODE using Lux
dudt = Lux.Chain(
    x -> x.^3,
    Lux.Dense(2, 50, tanh),
    Lux.Dense(50, 2)
)
ps, st = Lux.setup(rng, dudt)

n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat=t, abstol=1e-8, reltol=1e-6)

# Define the prediction function
function predict_n_ode(p)
    Array(n_ode(u0, p, st)[1])
end

# Define the loss function using the two-stage method
function loss_n_ode(p)
    pred = predict_n_ode(p)
    return sum(abs2, ode_data .- pred)
end

# Define the callback function
cb = function (p, l)
    println("Loss: ", l)
    pred = predict_n_ode(p)
    pl = scatter(t, ode_data[1, :], label="Data (u1)")
    scatter!(pl, t, pred[1, :], label="Prediction (u1)")
    plot!(pl, t, ode_data[2, :], label="Data (u2)")
    scatter!(pl, t, pred[2, :], label="Prediction (u2)")
    display(pl)
    return false
end

# Define the optimization problem
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_n_ode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(ps))

# Display initial ODE with initial parameters
cb(ps, loss_n_ode(ps))

# Train the neural ODE
res1 = Optimization.solve(optprob, ADAM(0.05); callback=cb, maxiters=100)
