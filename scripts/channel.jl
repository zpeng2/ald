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

# ╔═╡ c7b74ea0-5469-11eb-0ee4-73b2f96e3b07
@bind frame PlutoUI.Slider(0:99)

# ╔═╡ 0b93a760-5459-11eb-1359-5dbf4f8dd420
begin
	file = "U1.000tauR0.300.h5"
	y = Pda.get_h5data(file, "config/$frame/y")
	x = Pda.get_h5data(file, "config/$frame/x")
	theta = Pda.get_h5data(file, "config/$frame/theta")
end

# ╔═╡ 305b5bb0-5459-11eb-1d3e-21bbe247b5ef
loc, n = Pda.density1d(y, -0.5, 0.5)

# ╔═╡ aa8cb766-5469-11eb-0866-3dc01bb893a8
scatter(loc[2:end-1],n[2:end-1])

# ╔═╡ 9665ef90-546a-11eb-11b4-5dc0a130aba5
y1 = [val for val in y if abs(val) < 0.5]

# ╔═╡ a6747bf4-546a-11eb-17b7-8d2f885b4574
histogram(y1, label=nothing, framestyle=:box)

# ╔═╡ Cell order:
# ╠═fd8cab10-5458-11eb-3deb-dfba8e8d5152
# ╠═c7b74ea0-5469-11eb-0ee4-73b2f96e3b07
# ╟─0b93a760-5459-11eb-1359-5dbf4f8dd420
# ╟─305b5bb0-5459-11eb-1d3e-21bbe247b5ef
# ╠═aa8cb766-5469-11eb-0866-3dc01bb893a8
# ╠═9665ef90-546a-11eb-11b4-5dc0a130aba5
# ╠═a6747bf4-546a-11eb-17b7-8d2f885b4574
