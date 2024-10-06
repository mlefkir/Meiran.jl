struct BlockCovarianceMatrix
	Σ₁₁::Symmetric{Float64, Matrix{Float64}}
	# Σ₁₂::Matrix
	Σ₂₁::Matrix{Float64}
	Σ₂₂::Symmetric{Float64, Matrix{Float64}}
end

struct CholeskyBlockCovarianceMatrix
	L₁₁::LowerTriangular{Float64, Matrix{Float64}}
	L₂₁::Matrix{Float64}
	L₂₂::LowerTriangular{Float64, Matrix{Float64}}
end

function BlockMatrix_from_cs(cs::CrossSpectralDensity, t₁::Vector{Float64}, t₂::Vector{Float64}, σ_X₁²::Vector{Float64}, σ_X₂²::Vector{Float64}, f0::Float64, fM::Float64, J::Int64)

	ωⱼ = Vector{Float64}(undef, J)
	zⱼ = Vector{Float64}(undef, J)

	a_𝓟₁ = Vector{Float64}(undef, J)
	a_𝓟₂ = Vector{Float64}(undef, J)
	a_𝓒₁₂ = Vector{Float64}(undef, J)
	a_τ = Vector{Float64}(undef, J)

	approximate_cross_spectral_density!(ωⱼ, zⱼ, a_𝓟₁, a_𝓟₂, a_𝓒₁₂, a_τ, cs, f0, fM, J)

	Σ₁₁ = Matrix{Float64}(undef, length(t₁), length(t₁))
	Σ₂₂ = Matrix{Float64}(undef, length(t₂), length(t₂))
	Σ₂₁ = Matrix{Float64}(undef, length(t₂), length(t₁))

	for (i, t₁ᵢ) in enumerate(t₁)
		for (j, t₁ⱼ) in enumerate(t₁)
			Σ₁₁[i, j] = approximated_covariance(t₁ᵢ - t₁ⱼ, a_𝓟₁, ωⱼ, zⱼ, J)
			if i == j
				Σ₁₁[i, j] += σ_X₁²[i]
			end
		end
	end

	for (i, t₂ᵢ) in enumerate(t₂)
		for (j, t₂ⱼ) in enumerate(t₂)
			Σ₂₂[i, j] = approximated_covariance(t₂ᵢ - t₂ⱼ, a_𝓟₂, ωⱼ, zⱼ, J)
			if i == j
				Σ₂₂[i, j] += σ_X₂²[i]
			end
		end
		for (j, t₁ⱼ) in enumerate(t₁)
			Σ₂₁[i, j] = approximated_cross_covariance(t₂ᵢ - t₁ⱼ, a_𝓒₁₂, a_τ, ωⱼ, zⱼ, J)
		end
	end
	return BlockCovarianceMatrix(Symmetric(Σ₁₁), Σ₂₁, Symmetric(Σ₂₂))
end

function sanity_checks(Σ::BlockCovarianceMatrix)
	Σ₁₁, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₂₁, Σ.Σ₂₂

	# symmetry of Σ
	@assert isapprox(Σ₁₁, Σ₁₁') "Σ₁₁ is not symmetric"
	@assert isapprox(Σ₂₂, Σ₂₂') "Σ₂₂ is not symmetric"
	# @assert isapprox(Σ₁₂, Σ₂₁') "Σ₁₂ is not the transpose of Σ₂₁"

	# positive definiteness of Σ
	@assert isposdef(Σ₁₁) "Σ₁₁ is not positive definite"
	@assert isposdef(Σ₂₂) "Σ₂₂ is not positive definite"
	# @assert isposdef([Σ₁₁ Σ₁₂; Σ₂₁ Σ₂₂]) "Σ is not positive definite"
	schur = Symmetric(Σ₂₂ - Σ₂₁ * (Σ₁₁ \ (Σ₂₁')))
	@assert isposdef(schur) "Schur complement is not positive definite"
	@assert isposdef(inv(schur)) "Inverse of Schur complement is not positive definite"

end

"""
	cholesky(Σ::BlockCovarianceMatrix)

Compute the Cholesky decomposition of a block covariance matrix Σ. The matrix is decomposed as:

Σ = [L₁₁ 0; L₂₁ L₂₂] [L₁₁ 0; L₂₁ L₂₂]'

where L₁₁ is the Cholesky decomposition of the upper-left block of Σ, L₂₁ is the lower-left block of Σ, and L₂₂ is the Cholesky decomposition of the Schur complement of Σ.

# Arguments
- Σ::BlockCovarianceMatrix: the block covariance matrix to decompose

# Returns
- L₁₁::LowerTriangular{Float64,Array{Float64,2}}: the Cholesky decomposition of the upper-left block of Σ
- L₂₁::Matrix{Float64}: the lower-left block of Σ
- L₂₂::LowerTriangular{Float64,Array{Float64,2}}: the Cholesky decomposition of the Schur complement of Σ
"""
function LinearAlgebra.cholesky(Σ::BlockCovarianceMatrix)
	Σ₁₁, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₂₁, Σ.Σ₂₂
	L₁₁ = cholesky(Σ₁₁).L
	u = L₁₁ \ Σ₂₁'
	schur = Symmetric(Σ₂₂ - u' * u)
	L₂₂ = cholesky(schur).L
	L₂₁ = u'
	return L₁₁, L₂₁, L₂₂
end

function get_logdet_triangular(L)
	ldet = 0.0
	for i in eachindex(diag(L))
		ldet += 2log(L[i, i])
	end
	ldet
end

"""
	get_chi2term(L₁₁::LowerTriangular{Float64,Array{Float64,2}}, L₂₂::LowerTriangular{Float64,Array{Float64,2}}, Σ₂₁::Array{Float64,2}, x₁::Vector{Float64}, x₂::Vector{Float64})
	
"""
function get_chi2term(L₁₁::LowerTriangular{Float64, Array{Float64, 2}}, L₂₂::LowerTriangular{Float64, Array{Float64, 2}}, Σ₂₁::Array{Float64, 2}, x₁::Vector{Float64}, x₂::Vector{Float64})
	z₁ = L₁₁ \ x₁
	w = L₂₂ \ x₂
	v = L₁₁' \ z₁
	g = Σ₂₁ * v
	u = L₂₂ \ g
	z₂ = w - u
	return z₁' * z₁ + z₂' * z₂
end

function log_likelihood(cs::CrossSpectralDensity, t₁::Vector{Float64}, t₂::Vector{Float64}, x₁::Vector{Float64}, x₂::Vector{Float64}, σ²_x₁::Vector{Float64}, σ²_x₂::Vector{Float64}, f0::Float64, fM::Float64, J::Int)

	Σ = BlockMatrix_from_cs(cs, t₁, t₂, σ²_x₁, σ²_x₂, f0, fM, J)
	Σ₁₁, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₂₁, Σ.Σ₂₂
	L₁₁ = cholesky(Σ₁₁).L
	u = L₁₁ \ Σ₂₁'
	schur = Symmetric(Σ₂₂ - u' * u)
	L₂₂ = cholesky(schur).L

	return -0.5 * get_chi2term(L₁₁, L₂₂, Σ.Σ₂₁, x₁, x₂) - 0.5 * (get_logdet_triangular(L₁₁) + get_logdet_triangular(L₂₂))
end