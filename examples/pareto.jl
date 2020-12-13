### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6b732070-3cb2-11eb-15a0-65fd51108ed7
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end

# ╔═╡ d54000a4-3cc1-11eb-3eb9-e57ecd5f138b
@bind frame PlutoUI.Slider(0:499, show_value=true)

# ╔═╡ 550980e0-3ccb-11eb-0884-03a9d8e523f7
struct ConfigFrame{T}
	id::Int
	x::Vector{T}
	y::Vector{T}
	theta::Vector{T}
	t::T
	tauR::T
	function ConfigFrame(file::String, id::Int)
		t = Pda.get_h5data(file, "config/$(id)/t")
		x = Pda.get_h5data(file, "config/$(id)/x")
		y = Pda.get_h5data(file, "config/$(id)/y")
		theta = Pda.get_h5data(file, "config/$(id)/theta")
		T = eltype(x)
		tauR = Pda.get_h5attr(file, "tauR")
		return new{T}(id, x, y, theta, t, tauR)
	end
end

# ╔═╡ ac910d60-3cc1-11eb-03a7-cb017e973698
begin
	file = "lH1.000alpha1.200.h5"
	t = Pda.get_h5data(file, "config/$(frame)/t")
	y = Pda.get_h5data(file, "config/$(frame)/y")
	loc, n = Pda.density1d(y, bins=100, normalize=true)
	tauR = Pda.get_h5attr(file, "tauR")
end

# ╔═╡ 127bd560-3cc2-11eb-10d0-3f01f597307c
begin
	scatter(loc,n, framestyle=:box, label=nothing,yscale=:log10)
	xlabel!(L"y/H")
	ylabel!("number density")
	time_label = @sprintf "t/\\tau_R=%.2f" t/tauR
	title!(latexstring(time_label))
end

# ╔═╡ 92e63448-3ccc-11eb-1acf-3b4309d37ae2
begin
	plot(Pda.get_h5data(file,"x/t")/tauR,
		Pda.get_h5data(file, "x/m"),
		framestyle=:box,
		label=nothing)
	xlabel!(L"t/\tau_R")
	ylabel!(L"<x>")
end

# ╔═╡ edc909f8-3ccc-11eb-2a4c-31536c489723
begin
	plot(Pda.get_h5data(file,"x/t")/tauR,
		Pda.get_h5data(file, "x/v"),
		framestyle=:box,
		label=nothing)
	xlabel!(L"t/\tau_R")
	ylabel!(L"<x^2>")
end

# ╔═╡ Cell order:
# ╠═6b732070-3cb2-11eb-15a0-65fd51108ed7
# ╠═d54000a4-3cc1-11eb-3eb9-e57ecd5f138b
# ╟─550980e0-3ccb-11eb-0884-03a9d8e523f7
# ╠═ac910d60-3cc1-11eb-03a7-cb017e973698
# ╠═127bd560-3cc2-11eb-10d0-3f01f597307c
# ╟─92e63448-3ccc-11eb-1acf-3b4309d37ae2
# ╠═edc909f8-3ccc-11eb-2a4c-31536c489723
