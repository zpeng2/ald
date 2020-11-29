import ald
import numpy as np

particle = ald.RTP()
flow = ald.ZeroVelocity()
domain = ald.Box.from_freespace()

ic = ald.InitialConfig(
    x=ald.Point(0), y=ald.Uniform(-0.5, 0.5), theta=ald.Uniform(0, 2 * np.pi)
)


cfg = ald.Config(particle, domain, N=204800, dt=1e-4, Nt=1000000)


compiler = ald.RTPCompiler(particle, domain, flow, ic)

simulator = ald.Simulator(cfg, compiler)


# range to compute stats on configuration and print time.
stat_range = ald.InRange(start=0, stop=cfg.Nt, freq=50000)
# setup callbacks.
x = ald.MeanVariance(stat_range, "x", unwrap=True)
eta = ald.ETA(stat_range)
callbacks = [x, eta]

simulator.run(cfg, callbacks=callbacks)
