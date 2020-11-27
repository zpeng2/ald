from ald.core import particles
from ald.core.compiler import RTPCompiler
import ald

particle = ald.RTP()

box = ald.Box.from_freespace()

cfg = ald.Config(particle, box, N=204800, dt=1e-4, Nt=204800)

flow = ald.Poiseuille()

compiler = RTPCompiler(particle, box, flow)

simulator = ald.Simulator(cfg, compiler)


# range to compute stats on configuration and print time.
stat_range = ald.InRange(start=0, stop=cfg.Nt, freq=50000)
# setup callbacks.
stat_x = ald.MeanVariance(stat_range, "x", unwrap=True)
eta = ald.ETA(stat_range)
callbacks = [stat_x, eta]

simulator.run(cfg, callbacks=callbacks)
