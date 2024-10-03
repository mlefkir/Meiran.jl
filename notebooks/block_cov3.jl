function basis_cross_covariance(τ::Float64, a::Float64,Δτ::Float64, ω::Float64, z::Float64)
    return 2*a * ω * cos(2π * z * (τ+Δτ)) * sinc(ω * (τ+Δτ))
end
function basis_covariance(τ::Float64, a::Float64, ω::Float64, z::Float64)
return  2*a  * ω * cos(2π * z * τ ) * sinc(ω * τ)

end
function approximate_cross_spectral_density3!(
	ωⱼ::AbstractVector{Float64},
	zⱼ::AbstractVector{Float64},
	a_𝓟₁::AbstractVector{Float64},
	a_𝓟₂::AbstractVector{Float64},
	a_𝓒₁₂::AbstractVector{Float64},
	a_τ::AbstractVector{Float64},
	cs::CrossSpectralDensity,
	f0::Float64,
	fM::Float64,
	J::Int64,
)
	# first basis function centred at 0.
	ωⱼ[1] = 2 * f0#fⱼ[1]
	zⱼ[1] = 0.0

	q = (fM / f0)^(1.0 / (J - 1))
	# remaining basis functions
	for j in 2:J
		fⱼ, fⱼ₋₁ = f0 * q^(j - 1), f0 * q^(j - 2)
		ωⱼ[j] = fⱼ - fⱼ₋₁
		zⱼ[j] = fⱼ₋₁ + ωⱼ[j] / 2
	end

	a_𝓟₁[1] = cs.𝓟₁(f0) /2
	a_𝓟₂[1] = cs.𝓟₂(f0) /2 
	a_𝓒₁₂[1] = √a_𝓟₁[1] * √a_𝓟₂[1] 
	a_τ[1] = cs.Δφ(f0)
 

	zv = zⱼ[2:J]
	a_𝓟₁[2:end] = cs.𝓟₁.(zv) 
	a_𝓟₂[2:end] = cs.𝓟₂.(zv)
	a_𝓒₁₂[2:end] = @. √a_𝓟₁[2:end] * √a_𝓟₂[2:end]
	a_τ[2:end] = cs.Δφ(zv)
end
function BlockMatrix_from_cs_3(cs::CrossSpectralDensity, t₁::Vector{Float64}, t₂::Vector{Float64}, σ_X₁²::Vector{Float64}, σ_X₂²::Vector{Float64}, f0::Float64, fM::Float64, J::Int64)

	ωⱼ = Vector{Float64}(undef, J)
	zⱼ = Vector{Float64}(undef, J)

	a_𝓟₁ = Vector{Float64}(undef, J)
	a_𝓟₂ = Vector{Float64}(undef, J)
	a_𝓒₁₂ = Vector{Float64}(undef, J)
	a_τ = Vector{Float64}(undef, J)

	approximate_cross_spectral_density3!(ωⱼ, zⱼ, a_𝓟₁, a_𝓟₂, a_𝓒₁₂, a_τ, cs, f0, fM, J)

    Σ₁₁ = zeros(Float64,length(t₁), length(t₁))
    Σ₂₂ = zeros(Float64,length(t₂), length(t₂))
    Σ₂₁ = zeros(Float64,length(t₂), length(t₁))

	for l in 1:J
        ω = ωⱼ[l]
        z = zⱼ[l]

		for (i, t₁ᵢ) in enumerate(t₁)
			for (j, t₁ⱼ) in enumerate(t₁)
				Σ₁₁[i, j] += basis_covariance(t₁ᵢ - t₁ⱼ, a_𝓟₁[l], ω, z)
				if l == 1 &&i == j 
					Σ₁₁[i, j] += σ_X₁²[i]
				end
			end
		end

		for (i, t₂ᵢ) in enumerate(t₂)
			for (j, t₂ⱼ) in enumerate(t₂)
				Σ₂₂[i, j] += basis_covariance(t₂ᵢ - t₂ⱼ, a_𝓟₂[l], ω, z)
				if l == 1 && i == j
					Σ₂₂[i, j] += σ_X₂²[i]
				end
			end
			for (j, t₁ⱼ) in enumerate(t₁)
				Σ₂₁[i, j] += basis_cross_covariance(t₂ᵢ - t₁ⱼ, a_𝓒₁₂[l], a_τ[l],ω,z)
			end
		end
	end
	return BlockCovarianceMatrix(Symmetric(Σ₁₁), Σ₂₁, Symmetric(Σ₂₂))
end
Σ₃ = BlockMatrix_from_cs_3(cs, t, t, σ_x₁, σ_x₂, f0, fM, J)
Σ₁₁₃, Σ₂₁₃, Σ₂₂₃ = Σ₃.Σ₁₁, Σ₃.Σ₂₁, Σ₃.Σ₂₂