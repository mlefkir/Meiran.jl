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

	_, ωⱼ, zⱼ, a_𝓟₁, a_𝓟₂, a_𝓒₁₂, a_τ = approximate_cross_spectral_density(cs, f0, fM, J)

	d₁₁ = t₁ .- t₁'
	d₂₂ = t₂ .- t₂'
	d₁₂ = t₂ .- t₁'
	Σ₁₁ = Symmetric(approximated_covariance(d₁₁, a_𝓟₁, ωⱼ, zⱼ, J) + diagm(σ_X₁²))
	Σ₂₂ = Symmetric(approximated_covariance(d₂₂, a_𝓟₂, ωⱼ, zⱼ, J) + diagm(σ_X₂²))
	Σ₂₁ = approximated_cross_covariance(d₁₂, a_𝓒₁₂, a_τ, ωⱼ, zⱼ, J)
	# Σ₂₁ = Σ₁₂'

	return BlockCovarianceMatrix(Σ₁₁, Σ₂₁, Σ₂₂)
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

function LinearAlgebra.cholesky(Σ::BlockCovarianceMatrix)
	Σ₁₁, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₂₁, Σ.Σ₂₂
	schur = Symmetric(Σ₂₂ - Σ₂₁ * (Σ₁₁ \ (Σ₂₁')))
	L₁₁ = cholesky(Σ₁₁).L
	L₂₁ = Σ₂₁ / L₁₁'
	L₂₂ = cholesky(schur).L
	return L₁₁, L₂₁, L₂₂, schur
	#CholeskyBlockCovarianceMatrix(L₁₁, L₂₁, L₂₂)
end


function get_chi2term(L₁₁::LowerTriangular{Float64,Array{Float64,2}}, L₂₁::Array{Float64,2}, L₂₂::LowerTriangular{Float64,Array{Float64,2}}, Σ₂₁::Array{Float64,2}, x₁::Vector{Float64}, x₂::Vector{Float64})
	z₁ = L₁₁ \ x₁
	w = L₂₂ \ x₂
	v = L₁₁' \ z₁
	g = Σ₂₁ * v
	u = L₂₂ \ g
	z₂ = w - u
	return z₁' * z₁ + z₂' * z₂
end
function log_likelihood(cs::CrossSpectralDensity, t₁::Vector{Float64}, t₂::Vector{Float64}, x₁::Vector{Float64}, x₂::Vector{Float64}, σ_X₁²::Vector{Float64}, σ_X₂²::Vector{Float64}, f0::Float64, fM::Float64, J::Int)

	Σ = BlockMatrix_from_cs(cs, t₁, t₂, σ_X₁², σ_X₂², f0, fM, J)
	L₁₁, L₂₁, L₂₂, schur = cholesky(Σ)

	return -0.5 * get_chi2term(L₁₁, L₂₁, L₂₂, Σ.Σ₂₁, x₁, x₂) - 0.5 * (logdet(Σ.Σ₁₁) + logdet(schur))
end