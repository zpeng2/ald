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

# ╔═╡ 3f2a7e7c-7e03-11eb-3e84-316469e7d063
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end

# ╔═╡ 4713e678-7e03-11eb-3b02-bb688d4e9390
file = "./U5.000tauR1.000.h5"

# ╔═╡ 682d1eba-7e03-11eb-304f-e9733f0a44ce
tauR =1

# ╔═╡ 53f76e3c-7e03-11eb-187c-55c93b0ead58
begin
	plot(Pda.get_h5data(file,"x/t")/tauR,
		Pda.get_h5data(file, "x/m"),
		framestyle=:box,
		label=nothing)
	xlabel!(L"t/\tau_R")
	ylabel!(L"<x>")
end

# ╔═╡ 8714fc80-7e03-11eb-03f2-cd28a1f1af3a
begin
	plot(Pda.get_h5data(file,"x/t")/tauR,
		Pda.get_h5data(file, "x/v"),
		framestyle=:box,
		label=nothing,
		xscale=:log10,
		yscale=:log10)
	xlabel!(L"t/\tau_R")
	ylabel!("variance")
end

# ╔═╡ 0b0af80a-7e04-11eb-2351-090a0530f61b
 @bind frame PlutoUI.Slider(10:10:100, show_value = true)

# ╔═╡ 099e3036-7e04-11eb-36bc-fb3772b3578d
begin
	t = Pda.get_h5data(file, "config/$(frame)/t")
	y = Pda.get_h5data(file, "config/$(frame)/x")
	loc, n = Pda.density1d(y, bins=100, normalize=true)
end

# ╔═╡ 49247ae4-7e04-11eb-13f5-cbb82ebbd241
begin
	scatter(loc[2:end-1],n[2:end-1], framestyle=:box, label=nothing)#,yscale=:log10)
	xlabel!(L"y/H")
	ylabel!("number density")
	time_label = @sprintf "t/\\tau_R=%.2f" t/tauR
	title!(latexstring(time_label))
end

# ╔═╡ Cell order:
# ╠═3f2a7e7c-7e03-11eb-3e84-316469e7d063
# ╠═4713e678-7e03-11eb-3b02-bb688d4e9390
# ╠═682d1eba-7e03-11eb-304f-e9733f0a44ce
# ╠═53f76e3c-7e03-11eb-187c-55c93b0ead58
# ╠═8714fc80-7e03-11eb-03f2-cd28a1f1af3a
# ╠═0b0af80a-7e04-11eb-2351-090a0530f61b
# ╠═099e3036-7e04-11eb-36bc-fb3772b3578d
# ╠═49247ae4-7e04-11eb-13f5-cbb82ebbd241
