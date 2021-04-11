### A Pluto.jl notebook ###
# v0.12.18

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

# ╔═╡ 2ea55a90-7e03-11eb-2849-c52e0bfcc500


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

# ╔═╡ 442a778a-3d06-11eb-1f3f-b5d1e42ffaca
function jump_max(frame1, frame2)
	x0 = Pda.get_h5data(file, "config/$(frame1)/x")
	x1 = Pda.get_h5data(file, "config/$(frame2)/x")
	y0 = Pda.get_h5data(file, "config/$(frame1)/y")
	y1 = Pda.get_h5data(file, "config/$(frame2)/y")
	dx = sum((x1-x0).^2 +(y1-y0).^2)/length(x0)
	#dxm, i = findmax(dx)
	return dx 
end

# ╔═╡ d5f96e5a-3cfc-11eb-26de-414f6ec905a8
begin
	x1 = Float64[]
	y1 = Float64[]
	t2 =Float64[]
	i = 91466
	for frame in 0:199
			append!(x1, Pda.get_h5data(file, "config/$(frame)/x")[i])
			append!(y1, Pda.get_h5data(file, "config/$(frame)/y")[i])
			append!(t2, Pda.get_h5data(file, "config/$(frame)/t"))
	end
end

# ╔═╡ 4d58dede-3d04-11eb-03ef-69cfd329dbff
begin
	scatter(x1, y1, label=nothing, framestyle=:box)
	plot!(x1, y1, lw=2, label=nothing)
	xlabel!(L"x")
	ylabel!(L"y")
end

# ╔═╡ 535eb2b6-3d06-11eb-16b1-e3ebcc4e86f2
x1[end]-x1[1]

# ╔═╡ c36512f2-3d04-11eb-3afb-f3f2e62d07da
begin
	scatter(t2, x1, label=nothing, framestyle=:box)
	plot!(t2, x1, lw=2, label=nothing)
	xlabel!(L"t")
	ylabel!(L"x")
end

# ╔═╡ 127bd560-3cc2-11eb-10d0-3f01f597307c
begin
	scatter(loc[2:end-1],n[2:end-1], framestyle=:box, label=nothing)#,yscale=:log10)
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
	scatter(Pda.get_h5data(file,"x/t")[2:end]/tauR,
		Pda.get_h5data(file, "x/v")[2:end],
		framestyle=:box,
		label=nothing,
		xscale=:log10,
		yscale=:log10)
	t1 =Pda.get_h5data(file,"x/t")[2:end]/tauR
	v1 = Pda.get_h5data(file, "x/v")[2:end]
	plot!(t1, t1.^2/t1[1].^2*v1[1], linestyle=:dash, lw=2, label="slope=2")
	plot!(t1, t1.^1/t1[1].^1*v1[1],
		linestyle=:dash,
		lw=2,
		label="slope=1",
		legend=:topleft)
	xlabel!(L"t/\tau_R")
	ylabel!(L"<x^2>")
end

# ╔═╡ 4980b4d6-3d0a-11eb-2340-ed91b8805fc1
v1[751]/v1[748]

# ╔═╡ 2294b9ec-3d0a-11eb-1a1f-7bb34b614a9c
t1[750]

# ╔═╡ 43bdbdfe-3d09-11eb-3e64-497f67bf7e71
function pareto_runtime(U0, tauR)
	taum = (alpha - 1.0) * tauR / alpha
	return taum / pow(rand(), 1.0 / alpha)
end

# ╔═╡ f95566a2-3cfb-11eb-3202-bf60fca4728a
function pareto_update!(x::Vector{Float64},
		y::Vector{Float64},
		theta::Vector{Float64},
		U0::Real,
		tauR::Real,
		alpha::Real)
	
end

# ╔═╡ Cell order:
# ╠═6b732070-3cb2-11eb-15a0-65fd51108ed7
# ╠═2ea55a90-7e03-11eb-2849-c52e0bfcc500
# ╠═d54000a4-3cc1-11eb-3eb9-e57ecd5f138b
# ╠═442a778a-3d06-11eb-1f3f-b5d1e42ffaca
# ╠═d5f96e5a-3cfc-11eb-26de-414f6ec905a8
# ╠═4d58dede-3d04-11eb-03ef-69cfd329dbff
# ╠═535eb2b6-3d06-11eb-16b1-e3ebcc4e86f2
# ╠═c36512f2-3d04-11eb-3afb-f3f2e62d07da
# ╟─550980e0-3ccb-11eb-0884-03a9d8e523f7
# ╠═ac910d60-3cc1-11eb-03a7-cb017e973698
# ╠═127bd560-3cc2-11eb-10d0-3f01f597307c
# ╠═92e63448-3ccc-11eb-1acf-3b4309d37ae2
# ╠═edc909f8-3ccc-11eb-2a4c-31536c489723
# ╠═4980b4d6-3d0a-11eb-2340-ed91b8805fc1
# ╠═2294b9ec-3d0a-11eb-1a1f-7bb34b614a9c
# ╠═43bdbdfe-3d09-11eb-3e64-497f67bf7e71
# ╠═f95566a2-3cfb-11eb-3202-bf60fca4728a
