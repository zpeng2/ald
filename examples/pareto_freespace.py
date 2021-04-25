import ald
import numpy as np
import h5py

U0 = 1.0


alpha = 1.2
tauR = 1.0
L = 1.0  # simulation box dimension.
particle = ald.Pareto(U0=U0, tauR=tauR)

flow = ald.ZeroVelocity()
domain = ald.Box(left=-L / 2, right=L / 2, bottom=-L / 2, top=L / 2)

ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)

# number of particles
N = 300_000
dt = 1e-4
Nt = 40_000_000
cfg = ald.Config(particle, domain, N=N, dt=dt, Nt=Nt)

kernel = ald.RTPFreespaceKernel()
compiler = ald.RTPCompiler(kernel, cfg, flow, ic)

compiler.compile()

simulator = ald.RTPSimulator(cfg, compiler)

file = "U{:.3f}alpha{:.3f}free.h5".format(U0, alpha)

# create an empty file
with h5py.File(file, "w") as f:
    pass
# range to compute stats on configuration and print time.
runner = ald.RangedRunner(start=0, stop=cfg.Nt, freq=10000)
# setup callbacks.
x = ald.DisplacementMeanVariance(runner, "x", unwrap=True)
y = ald.DisplacementMeanVariance(runner, "y", unwrap=True)
# y = ald.MeanVariance(runner, "y", unwrap=True)
callbacks = [x, y, ald.ETA(runner)]

simulator.run(cfg, callbacks=callbacks)

# save particle, domain and simulation attributes.
cfg.save2h5(file)
# save mean variance of x
x.save2h5(file, "x")
y.save2h5(file, "y")
