import ald
import numpy as np


particle = ald.ABP(U0=1.0, DT=1.0, tauR=1.0)

flow = ald.ZeroVelocity()

domain = ald.Box(left=0, right=1, bottom=-0.5, top=0.5)

ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)


cfg = ald.Config(particle, domain, N=204800, dt=1e-4, Nt=2000000)

kernel = ald.PlanarWallKernel(cfg)


compiler = ald.ABPCompiler(kernel, cfg, flow, ic)

simulator = ald.ABPSimulator(cfg, compiler)


runner = ald.RangedRunner(start=0, stop=cfg.Nt, freq=10000)
# setup callbacks.
file = "abp.h5"
configsaver = ald.ConfigSaver(runner, file, variables=["x", "y", "theta"])
callbacks = [ald.ETA(runner), configsaver]


simulator.run(cfg, callbacks=callbacks)
