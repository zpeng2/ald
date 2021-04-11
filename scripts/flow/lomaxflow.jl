### A Pluto.jl notebook ###
# v0.14.1

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

# ╔═╡ 7fa4e0c6-9a93-11eb-3c27-b948ad182e72
begin
	using HDF5
	using Plots
	using PlutoUI
	using LaTeXStrings
	using Printf
	import Pda
end


# ╔═╡ ca6d567e-db4a-4d92-b7f3-1d0dc279d1b9
file = "./lH1.000Peg1.000alpha1.200.h5"

# ╔═╡ 28d9e092-8bcf-436c-8b62-678710b64759
@bind frame PlutoUI.Slider(0:99, show_value=true)

# ╔═╡ 62bde565-8ac4-4af4-834d-d2b716b9be50
begin
	x = Pda.get_h5data(file, "config/$frame/x")
	y = Pda.get_h5data(file, "config/$frame/y")
end

# ╔═╡ e5c924fa-7fc1-4205-ab90-c5dc66b128dc
ybin, n = Pda.density1d(y, bins=100)

# ╔═╡ b1d20efe-8e8e-44ca-8df3-9b24e5ebac2b
scatter(ybin[2:end-1],n[2:end-1])

# ╔═╡ 9c1091f3-27b0-460e-b3b0-135c81cae84f
HDF5.h5open(file, "r")

# ╔═╡ 17258ca3-14c6-4a2b-8f16-7fe746bae327
begin
	tm = Pda.get_h5data(file, "x/t")
	xm = Pda.get_h5data(file, "x/m")
	xv = Pda.get_h5data(file, "x/v")
end

# ╔═╡ 9a9fdce3-7dbb-4c83-9b89-7336c2eb6814
plot(tm, xv, framestyle = :box, label =nothing)

# ╔═╡ 5b8de4e3-ed90-4c5d-8981-41956d0b602a
plot(tm, xm, framestyle = :box, label =nothing)

# ╔═╡ c97b8438-47a0-4e62-8894-205393818fcf
begin
	tfrac = Pda.get_h5data(file, "wallfraction/t")
	wallfrac = Pda.get_h5data(file, "wallfraction/f")
	scatter(tfrac, wallfrac, framestyle = :box, label =nothing)
end

# ╔═╡ Cell order:
# ╠═7fa4e0c6-9a93-11eb-3c27-b948ad182e72
# ╠═ca6d567e-db4a-4d92-b7f3-1d0dc279d1b9
# ╠═28d9e092-8bcf-436c-8b62-678710b64759
# ╠═62bde565-8ac4-4af4-834d-d2b716b9be50
# ╠═e5c924fa-7fc1-4205-ab90-c5dc66b128dc
# ╠═b1d20efe-8e8e-44ca-8df3-9b24e5ebac2b
# ╠═9c1091f3-27b0-460e-b3b0-135c81cae84f
# ╠═17258ca3-14c6-4a2b-8f16-7fe746bae327
# ╠═9a9fdce3-7dbb-4c83-9b89-7336c2eb6814
# ╠═5b8de4e3-ed90-4c5d-8981-41956d0b602a
# ╠═c97b8438-47a0-4e62-8894-205393818fcf
