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

# ╔═╡ fd8cab10-5458-11eb-3deb-dfba8e8d5152
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end

# ╔═╡ 8fa90742-5522-11eb-230f-89826748854e
file = "../examples/U1.000tauR0.2001D.h5"

# ╔═╡ c7b74ea0-5469-11eb-0ee4-73b2f96e3b07
@bind frame PlutoUI.Slider(0:99, show_value=true)

# ╔═╡ b336edc8-5522-11eb-009c-717fef97aff1
begin
	x = Pda.get_h5data(file, "config/$frame/x")
	q = Pda.get_h5data(file, "config/$frame/direction")
end

# ╔═╡ 5a678c92-55ff-11eb-1300-b1148bd3b9c8
begin
	bins=100
	gridx = LinRange(-0.5, 0.5, bins+1)
	bins = length(gridx) -1
    # initialize array to store number
    m = zeros(bins)
    # in a cell [a,b], particles that has a<=x<b is counted.
    # for the last cell, x==b is also counted.
    for i in 1:bins
        for (position, localq) in zip(x,q)
            if gridx[i] <=position <gridx[i+1]
                m[i] += localq
            end
        end
    end
    # last cell: add particles on the right boundary
	#m[end] += sum(x .== gridx[end])
    # use bin centers as representative locations.
    dx = diff(gridx)
    bin_centers = gridx[1:end-1] + dx/2
    # if normalize
    #     n ./= dx *length(x)
    # end
end

# ╔═╡ 305b5bb0-5459-11eb-1d3e-21bbe247b5ef
scatter(bin_centers[2:end-1], m[2:end-1],framestyle=:box,label=nothing)

# ╔═╡ Cell order:
# ╠═fd8cab10-5458-11eb-3deb-dfba8e8d5152
# ╠═8fa90742-5522-11eb-230f-89826748854e
# ╠═c7b74ea0-5469-11eb-0ee4-73b2f96e3b07
# ╠═b336edc8-5522-11eb-009c-717fef97aff1
# ╟─5a678c92-55ff-11eb-1300-b1148bd3b9c8
# ╠═305b5bb0-5459-11eb-1d3e-21bbe247b5ef
