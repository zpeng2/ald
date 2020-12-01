import ald
import numpy as np

particle = ald.ABP()
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


# range to compute stats on configuration and print time.
stat_range = ald.InRange(start=0, stop=cfg.Nt, freq=10000)
# setup callbacks.

callbacks = [ald.ETA(stat_range)]

simulator.run(cfg, callbacks=callbacks)
