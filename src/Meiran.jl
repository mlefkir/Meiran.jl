module Meiran
path_to_package = "~/github/Pioran.jl"
push!(LOAD_PATH, path_to_package)
using Tonari
using LinearAlgebra

include("approximate.jl")
include("block_covmatrix.jl")

export approximate_cross_spectral_density,
	approximated_cross_covariance,
	approximated_covariance,
	BlockCovarianceMatrix,
	CholeskyBlockCovarianceMatrix,BlockMatrix_from_cs,
	sanity_checks, cholesky
end
