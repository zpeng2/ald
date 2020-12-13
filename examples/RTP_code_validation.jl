### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ a4eba47e-3cde-11eb-2a1c-7f5102c9a9e1
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end

# ╔═╡ b0c18c94-3cde-11eb-0c12-731d453f428b
md"""## Exponential RTP in free space"""

# ╔═╡ f2663b88-3cdf-11eb-2869-5f3d5ff56948
md"""The MSD of exponential RTPs with complete reorientation and initial condition $$P(\mathbf{x}, \theta,t) = \delta(\mathbf{x})/2\pi$$ is 

$$\left<x^2\right> =\left<y^2\right>=  U^2_0\tau^2_R\left( \frac{t}{\tau_R} + \exp(-t/\tau_R) -1\right)$$

Ref: [Run-and-tumble bacteria slowly approaching the diffusive regime](https://doi.org/10.1103/PhysRevE.101.062607), Villa-Torrealba *et. al*, PRE, 2020
"""

# ╔═╡ 3635fca6-3ce0-11eb-0d4b-1310b1716b86
function msd_exponential(t::Real, U0::Real, tauR::Real)
	return U0^2*tauR^2*(t/tauR+exp(-t/tauR) -1)
end

# ╔═╡ 8b42ee14-3ce2-11eb-0702-5da8b9887bd9
function get_var(file::String)
	t = Pda.get_h5data(file, "x/t")[2:end]
	v = Pda.get_h5data(file, "x/v")[2:end]
	tauR = Pda.get_h5attr(file, "tauR")
	U0 = Pda.get_h5attr(file, "U0")
	ell = U0*tauR
	# non-dimensionalized by tauR and ell
	t ./= tauR
	v ./= ell^2
	return t,v
end

# ╔═╡ d0f3bf32-3cde-11eb-13c0-6d1d3bd0b070
begin
	file = "U1.000tauR1.200.h5"
	t,v = get_var(file)
	plot(t,v,
		framestyle=:box,
		label="Langevin",
		lw=2)
	plot!(t, msd_exponential.(t, 1, 1)/1^2,
		lw=2,
		linestyle=:dash,
		label="theory",
		legend=:topleft,
		xscale=:log10,
		yscale=:log10)
	xlabel!(L"t/\tau_R")
	ylabel!(L"\mathrm{var}(x)/\ell^2")
	title!("Exponential RTP in free space")
end

# ╔═╡ 48f91100-3ce2-11eb-1331-930d15c04e16
md"""## Pareto RTP in free space"""

# ╔═╡ 53054a04-3ce2-11eb-2869-973ecb35edeb
begin
	file1= "U1.000alpha1.200free.h5"
	t1,v1 = get_var(file1)
	plot(t1,v1,
		framestyle=:box,
		label="Langevin",
		lw=2)
	alpha = Pda.get_h5attr(file1, "alpha")
	# #theory is <x^2> \sim t^(3-\alpha)
	i = 3000
	plot!(t1[end-i:end], (t1[end-i:end]/t1[end]).^(3-alpha)*v1[end],
		lw=2,
		linestyle=:dash,
		label="theory",
		legend=:topleft,
		xscale=:log10,
		yscale=:log10)
	xlabel!(L"t/\tau_R")
	ylabel!(L"\mathrm{var}(x)/\ell^2")
	title!("Pareto RTP in free space")
end

# ╔═╡ Cell order:
# ╠═a4eba47e-3cde-11eb-2a1c-7f5102c9a9e1
# ╟─b0c18c94-3cde-11eb-0c12-731d453f428b
# ╟─f2663b88-3cdf-11eb-2869-5f3d5ff56948
# ╠═3635fca6-3ce0-11eb-0d4b-1310b1716b86
# ╠═8b42ee14-3ce2-11eb-0702-5da8b9887bd9
# ╠═d0f3bf32-3cde-11eb-13c0-6d1d3bd0b070
# ╟─48f91100-3ce2-11eb-1331-930d15c04e16
# ╠═53054a04-3ce2-11eb-2869-973ecb35edeb
