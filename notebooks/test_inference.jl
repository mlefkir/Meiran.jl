#= Example script for the inference of a single bending power spectral density using Pioran and ultranest (python package).

run with:

julia single.jl data.txt

where `data.txt` is a file containing the time series data. The file should have three columns: time, flux, flux error. 
The script will create a directory `inference` containing the results of the inference.

If you have MPI installed, you may want to run the script in parallel, using the following command:

mpirun -n 4 julia single.jl data.txt

where `-n 4` is the number of processes to use.

=#

# load MPI and initialise
using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using Meiran
using Tonari
using Distributions
using Random
using DelimitedFiles
using Statistics
using PyCall
# load the python package ultranest
ultranest = pyimport("ultranest")
# define the random number generator
rng = MersenneTwister(123)

# get the filename from the command line
filename = ARGS[1]
fname = replace(split(filename, "/")[end], ".txt" => "_single")
dir = "inference/" * fname
plot_path = dir * "/plots/"

# Load the data and extract a subset for the analysis
A = readdlm(filename, comments = true, comment_char = '#')
t, x₁, x₂, σ_x₁, σ_x₂ = A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]

x₁ .-= mean(x₁)
x₂ .-= mean(x₂)
# Frequency range for the approx and the prior
f_min, f_max = 1 / (t[end] - t[1]), 1 / minimum(diff(t)) / 2

# options for the approximation
S_low, S_high = 5, 5
f0, fM = 1 / t[end] / S_low, 1 / (2 * minimum(diff(t))) * S_high
min_f_b, max_f_b = f0 * 4.0, fM / 4.0

n_components = 20
model = SingleBendingPowerLaw

function loglikelihood(x₁::Vector{Float64}, x₂::Vector{Float64}, t₁::Vector{Float64}, t₂::Vector{Float64}, σ_x₁::Vector{Float64}, σ_x₂::Vector{Float64}, params::Vector{Float64})
	θ₁, θ₂, Δτ = params[1:3], params[4:6], params[7]

	# Define power spectral density function
	𝓟₁ = model(θ₁...)
	𝓟₂ = model(θ₂...)
	Δϕ = ConstantTimeLag(Δτ)
	𝓒₁₂ = CrossSpectralDensity(𝓟₁, 𝓟₂, Δϕ)

	return log_likelihood(𝓒₁₂, t₁, t₂, x₁, x₂, σ_x₁, σ_x₂, f0, fM, n_components)
end
logl(pars::Vector{Float64}) = loglikelihood(x₁, x₂, t, t, σ_x₁ .^ 2, σ_x₂ .^ 2, pars)

# # Priors
function prior_transform(cube)
	α₁¹ = quantile(Uniform(0.0, 1.5), cube[1])
	f₁¹ = quantile(LogUniform(min_f_b, max_f_b), cube[2])
	α₂¹ = quantile(Uniform(1.5, 4.0), cube[3])
	α₁² = quantile(Uniform(0.0, 1.5), cube[4])
	f₁² = quantile(LogUniform(min_f_b, max_f_b), cube[5])
	α₂² = quantile(Uniform(1.5, 4.0), cube[6])
	Δτ = quantile(Uniform(-t[end] / 4, t[end] / 4), cube[7])
	return [α₁¹, f₁¹, α₂¹, α₁², f₁², α₂², Δτ]
end
paramnames = ["α₁¹", "f₁¹", "α₂¹", "α₁²", "f₁²", "α₂²", "Δτ"]

# println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

# println("Running sampler...")
sampler = ultranest.ReactiveNestedSampler(paramnames, logl, resume = true, transform = prior_transform, log_dir = dir, vectorized = false)
results = sampler.run()
sampler.print_results()
sampler.plot()
