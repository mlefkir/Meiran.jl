using Revise
using LinearAlgebra
using Tonari
using Plots
using Random
using Meiran
using BenchmarkTools
using ProfileView

rng = MersenneTwister(1234);

p1 = SingleBendingPowerLaw(1.0, 0.30, 4e-2, 3.5)
p2 = SingleBendingPowerLaw(1.0, 0.550, 8e-2, 2.95)
Δϕ = ConstantTimeLag(5.4)
cs = CrossSpectralDensity(p1, p2, Δϕ)

T = 100.0
Δt = 1.0

simu = Simulation(cs, T, Δt)
t, x, xerr = sample(rng,simu,1,error_size=0.1)
x₁,x₂ = x[1][:],x[2][:]
σ_x₁, σ_x₂ = xerr[1][:], xerr[2][:]

scatter(t,x₁,yerr=σ_x₁,label="X₁")
scatter!(t,x₂,yerr=σ_x₂,label="X₂")

S_low, S_high = 5, 5
f0, fM = 1 / T / S_low, 1 / (2Δt) * S_high
J = 50

log_likelihood(cs, t, t, x₁, x₂, σ_x₁.^2, σ_x₂.^2, f0, fM, J)

using ProfileView
VSCodeServer.@profview log_likelihood(cs, t, t, x₁, x₂, σ_x₁.^2, σ_x₂.^2, f0, fM, J)