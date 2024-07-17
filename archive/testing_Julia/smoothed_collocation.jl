using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, SciMLSensitivity, Optimization,
      OptimizationOptimisers, Plots

using Random
rng = Xoshiro(0)

u0 = Float32[2.0; 0.0]
datasize = 300
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps)) .+ 0.1randn(2, 300)

du, u = collocate_data(data, tsteps, EpanechnikovKernel())

scatter(tsteps, data')
plot!(tsteps, u'; lw = 5)
savefig("colloc.png")
plot(tsteps, du')
savefig("colloc_du.png")

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))

function loss(p)
    cost = zero(first(p))
    for i in 1:size(du, 2)
        _du, _ = dudt2(@view(u[:, i]), p, st)
        dui = @view du[:, i]
        cost += sum(abs2, dui .- _du)
    end
    sqrt(cost)
end

pinit, st = Lux.setup(rng, dudt2)

callback = function (p, l)
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(pinit))

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback, maxiters = 10000)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)
nn_sol, st = prob_neuralode(u0, result_neuralode.u, st)
scatter(tsteps, data')
plot!(nn_sol)
savefig("colloc_trained.png")

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(pinit))

numerical_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback, maxiters = 300)

nn_sol, st = prob_neuralode(u0, numerical_neuralode.u, st)
scatter(tsteps, data')
plot!(nn_sol; lw = 5)