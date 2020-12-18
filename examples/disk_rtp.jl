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

# ╔═╡ b88b727a-3f3e-11eb-3e3c-bb62b380a257
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end

# ╔═╡ 92ae91ea-3f52-11eb-12f9-497e9140820d
@bind frame PlutoUI.Slider(0:99, show_value=true)

# ╔═╡ a6760a18-3f5d-11eb-3cac-152259f7d4c8
file = "U1.000tauR1.000disk.h5"

# ╔═╡ c2d093dc-3f3e-11eb-187e-195e52e1fd31
begin
	x = Pda.get_h5data(file, "config/$(frame)/x")
	y = Pda.get_h5data(file, "config/$(frame)/y")
end

# ╔═╡ d2bd59f4-3f5c-11eb-3201-9fdd56f3174b
function average_radial_density(file)
	bins =100
	loc = zeros(bins)
	n = zeros(bins)
	frames = 100
	for frame in 0:frames-1
		x = Pda.get_h5data(file, "config/$(frame)/x")
		y = Pda.get_h5data(file, "config/$(frame)/y")
		r = sqrt.(x.^2 + y.^2)
		loc, n_ = Pda.density1d(r, 0, maximum(r), bins=100, normalize=false)
		n += n_
	end
	return loc, n/frames
end

# ╔═╡ 13220fe6-3f5d-11eb-3b9a-3d925df3f57c
loc,n = average_radial_density(file)

# ╔═╡ 18248b9a-3f3f-11eb-018f-cdf6768c2ac4
scatter(loc[2:end-1], (n./loc)[2:end-1], framestyle=:box, label=nothing)

# ╔═╡ Cell order:
# ╠═b88b727a-3f3e-11eb-3e3c-bb62b380a257
# ╠═92ae91ea-3f52-11eb-12f9-497e9140820d
# ╠═a6760a18-3f5d-11eb-3cac-152259f7d4c8
# ╠═c2d093dc-3f3e-11eb-187e-195e52e1fd31
# ╠═d2bd59f4-3f5c-11eb-3201-9fdd56f3174b
# ╠═13220fe6-3f5d-11eb-3b9a-3d925df3f57c
# ╠═18248b9a-3f3f-11eb-018f-cdf6768c2ac4
