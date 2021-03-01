import ald
import numpy as np
import h5py

U0 = 1
tauR = 0.2
particle = ald.Pareto(U0=U0, tauR=tauR)


# still using 2D interface
flow = ald.ZeroVelocity()
domain = ald.Box(left=-0.5, right=0.5)


ic = ald.InitialConfig(x=ald.Uniform(domain.left, domain.right))

N = 400_000
dt = 1e-4
Nt = 4_000_000
cfg = ald.Config(particle, domain, N=N, dt=dt, Nt=Nt)


# Use the 1D kernel, this is the only difference basically.
kernel = ald.Confined1DRTPKernel()

compiler = ald.RTP1DCompiler(kernel, cfg, flow, ic)

compiler.compile()

simulator = ald.RTP1DSimulator(cfg, compiler)

file = "U{:.3f}tauR{:.3f}1D.h5".format(U0, tauR)
# create an empty file
# with h5py.File(file, "w") as f:
#     pass
# range to compute stats on configuration and print time.
runner = ald.RangedRunner.from_backward_count(stop=cfg.Nt, freq=10000, count=50)
configsaver = ald.ConfigSaver(
    runner, file, variables=["x", "direction"], unwrap=[False, False]
)
# y = ald.MeanVariance(runner, "y", unwrap=True)
eta = ald.ETA(ald.RangedRunner(start=0, stop=cfg.Nt, freq=50000))
callbacks = [configsaver, eta]


simulator.run(cfg, callbacks=callbacks)

# save particle, domain and simulation attributes.
cfg.save2h5(file)
