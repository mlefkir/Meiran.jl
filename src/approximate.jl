using Tonari

""" Approximate the cross-spectral density with a sum J of top-hats between frequencies f0 and fM.

	# Arguments
	- `cs::CrossSpectralDensity`: the cross-spectral density to approximate.
	- `f0::Float64`: the lower frequency bound.
	- `fM::Float64`: the upper frequency bound.
	- `J::Int64`: the number of top-hats to use.

	# Returns
	- `fⱼ::Vector{Float64}`: the limiting frequencies of the top-hats.
	- `ωⱼ::Vector{Float64}`: the widths of the top-hats.
	- `zⱼ::Vector{Float64}`: the centres of the top-hats.
	- `a_𝓟₁::Vector{Float64}`: the power spectral density amplitude of the first channel.
	- `a_𝓟₂::Vector{Float64}`: the power spectral density amplitude of the second channel
	- `a_𝓒₁₂::Vector{Float64}`: the cross-spectral density amplitude.
	- `a_τ::Vector{Float64}`: the time delay amplitude.
"""
function approximate_cross_spectral_density(cs::CrossSpectralDensity, f0::Float64, fM::Float64, J::Int64)

	fⱼ = 10 .^ range(log10(f0), log10(fM), length = J)
	ωⱼ = Vector{Float64}(undef, J)
	zⱼ = Vector{Float64}(undef, J)

	# first basis function centred at 0.
	ωⱼ[1] = 2fⱼ[1]
	zⱼ[1] = 0.0

	# remaining basis functions
	for j in 2:J
		ωⱼ[j] = (fⱼ[j] - fⱼ[j-1])
		zⱼ[j] = fⱼ[j-1] + ωⱼ[j] / 2
	end

	a_𝓟₁ = Vector{Float64}(undef, J)
	a_𝓟₂ = Vector{Float64}(undef, J)
	a_𝓒₁₂ = Vector{Float64}(undef, J)
	a_τ = Vector{Float64}(undef, J)

	a_𝓟₁[1] = cs.𝓟₁(fⱼ[1])
	a_𝓟₂[1] = cs.𝓟₂(fⱼ[1])
	a_𝓒₁₂[1] = √a_𝓟₁[1] * √a_𝓟₂[1]
	a_τ[1] = cs.Δφ(fⱼ[1]) 

	a_𝓟₁[2:end] = cs.𝓟₁.(zⱼ[2:end])
	a_𝓟₂[2:end] = cs.𝓟₂.(zⱼ[2:end])
	a_𝓒₁₂[2:end] = .√a_𝓟₁[2:end] .* .√a_𝓟₂[2:end]
	a_τ[2:end] = cs.Δφ(zⱼ[2:end]) 

	return fⱼ, ωⱼ, zⱼ, a_𝓟₁, a_𝓟₂, a_𝓒₁₂, a_τ
end


""" Approximate the covariance function with a sum of J sinusoids (top-hats in the Fourier domain) 

We assume the first top-hat in the Fourier domain is centred on 0. 

# Arguments
- `τ::Vector{Float64}`: Time lags
- `aⱼ::Vector{Float64}`: Amplitudes of the basis functions
- `ωⱼ::Vector{Float64}`: Widths of the basis functions
- `zⱼ::Vector{Float64}`: Phase shifts of the basis functions
- `J::Int64`: Number of basis functions

# Returns
- `R::Matrix{Float64}`: Approximated covariance function
"""
function approximated_covariance(τ, aⱼ, ωⱼ, zⱼ, J)
	R = @. aⱼ[1] * ωⱼ[1] * cos(2π * zⱼ[1] * τ) * sinc(ωⱼ[1] * τ)
	for j in 2:J
		R += @. 2aⱼ[j] * ωⱼ[j] * cos(2π * zⱼ[j] * τ) * sinc(ωⱼ[j] * τ)
	end
	return R
end

""" Approximate the cross-covariance function with a sum of J sinusoids (top-hats in the Fourier domain)

We assume the first top-hat in the Fourier domain is centred on 0.

# Arguments
- `τ::Vector{Float64}`: Time lags
- `aⱼ::Vector{Float64}`: Amplitudes of the basis functions
- `τⱼ::Vector{Float64}`: Amplitude of the time delay function
- `ωⱼ::Vector{Float64}`: Widths of the basis functions
- `zⱼ::Vector{Float64}`: Phase shifts of the basis functions
- `J::Int64`: Number of basis functions

# Returns
- `R::Matrix{Float64}`: Approximated cross-covariance function
"""
function approximated_cross_covariance(τ, aⱼ, τⱼ, ωⱼ, zⱼ, J)
	R = @. aⱼ[1] * ωⱼ[1] * cos(2π * zⱼ[1] * (τ + τⱼ[1])) * sinc(ωⱼ[1] * (τ + τⱼ[1]))
	for j in 2:J
		R += @. 2aⱼ[j] * ωⱼ[j] * cos(2π * zⱼ[j] * (τ + τⱼ[j])) * sinc(ωⱼ[j] * (τ + τⱼ[j]))
	end
	return R
end