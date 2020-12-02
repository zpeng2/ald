import ald
import numpy as np

particle = ald.RTP(U0=1, tauR=1.0)
flow = ald.ZeroVelocity()
domain = ald.Box.from_freespace(Lx=1.0, Ly=1.0)

ic = ald.InitialConfig(
    x=ald.Point(0), y=ald.Uniform(-0.5, 0.5), theta=ald.Uniform(0, 2 * np.pi)
)


cfg = ald.Config(particle, domain, N=204800, dt=1e-4, Nt=2000000)


compiler = ald.RTPCompiler(cfg, flow, ic)

simulator = ald.RTPSimulator(cfg, compiler)


# range to compute stats on configuration and print time.
runner = ald.RangedRunner(start=0, stop=cfg.Nt, freq=10000)
# setup callbacks.
x = ald.MeanVariance(runner, "x", unwrap=True)
y = ald.MeanVariance(runner, "y", unwrap=True)
# y = ald.MeanVariance(runner, "y", unwrap=True)
callbacks = [x, y, ald.ETA(runner)]

simulator.run(cfg, callbacks=callbacks)
