struct BlockCovarianceMatrix
    Σ₁₁::Matrix
    Σ₁₂::Matrix
    Σ₂₁::Matrix
    Σ₂₂::Matrix
end

struct CholeskyBlockCovarianceMatrix
    L₁₁::Matrix
    L₂₁::Matrix
    L₂₂::Matrix
end

function BlockMatrix_from_cs(cs::CrossSpectralDensity, t, σ_X₁², σ_X₂², f0::Float64, fM::Float64, J::Int64)

	_, ωⱼ, zⱼ, a_𝓟₁, a_𝓟₂, a_𝓒₁₂, a_τ = approximate_cross_spectral_density(cs, f0, fM, J)

	dt = t .- t'
	d = abs.(dt)
	Σ₁₁ = approximated_covariance(d, a_𝓟₁, ωⱼ, zⱼ, J) + diagm(σ_X₁²)
	Σ₂₂ = approximated_covariance(d, a_𝓟₂, ωⱼ, zⱼ, J) + diagm(σ_X₂²)
	Σ₁₂ = approximated_cross_covariance(dt, a_𝓒₁₂, a_τ, ωⱼ, zⱼ, J)
	Σ₂₁ = Σ₁₂'

	return BlockCovarianceMatrix(Σ₁₁, Σ₁₂, Σ₂₁, Σ₂₂)
end

function sanity_checks(Σ::BlockCovarianceMatrix)
    Σ₁₁, Σ₁₂, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₁₂, Σ.Σ₂₁, Σ.Σ₂₂
    
    # symmetry of Σ
    @assert isapprox(Σ₁₁, Σ₁₁') "Σ₁₁ is not symmetric"
    @assert isapprox(Σ₂₂, Σ₂₂') "Σ₂₂ is not symmetric"
    @assert isapprox(Σ₁₂, Σ₂₁') "Σ₁₂ is not the transpose of Σ₂₁"

    # positive definiteness of Σ
    @assert isposdef(Σ₁₁) "Σ₁₁ is not positive definite"
    @assert isposdef(Σ₂₂) "Σ₂₂ is not positive definite"
    # @assert isposdef([Σ₁₁ Σ₁₂; Σ₂₁ Σ₂₂]) "Σ is not positive definite"
    schur = Symmetric(Σ₂₂ - (Σ₁₂' * inv(Σ₁₁) * Σ₁₂))
    @assert isposdef(schur) "Schur complement is not positive definite"
    @assert isposdef(inv(schur)) "Inverse of Schur complement is not positive definite"

end

function LinearAlgebra.cholesky(Σ::BlockCovarianceMatrix)
    Σ₁₁, Σ₁₂, Σ₂₁, Σ₂₂ = Σ.Σ₁₁, Σ.Σ₁₂, Σ.Σ₂₁, Σ.Σ₂₂
    schur = Symmetric(Σ₂₂ - (Σ₁₂' * inv(Σ₁₁) * Σ₁₂))
    L₁₁ = cholesky(Σ₁₁).L
    L₂₁ = Σ₂₁ * inv(L₁₁)'
    L₂₂ = cholesky(schur).L
    return L₁₁, L₂₁, L₂₂
    #CholeskyBlockCovarianceMatrix(L₁₁, L₂₁, L₂₂)
end
