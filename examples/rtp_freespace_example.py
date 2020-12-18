import ald
import numpy as np
import h5py

U0 = 1.0
tauR = 1.2
particle = ald.ExponentialRTP(U0=U0, tauR=tauR)

flow = ald.ZeroVelocity()
domain = ald.Box()


ic = ald.InitialConfig(
    x=ald.Point((domain.left + domain.right) / 2),
    y=ald.Point((domain.bottom + domain.top) / 2),
    theta=ald.Uniform(0, 2 * np.pi),
)


cfg = ald.Config(particle, domain, N=204800, dt=1e-4, Nt=4_000_000)

kernel = ald.RTPFreespaceKernel()
compiler = ald.RTPCompiler(kernel, cfg, flow, ic)

compiler.compile()

simulator = ald.RTPSimulator(cfg, compiler)

file = "U{:.3f}tauR{:.3f}.h5".format(U0, tauR)
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
