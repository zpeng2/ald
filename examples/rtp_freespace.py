from ald.rtp import callbacks
import ald

rtp = ald.RTP()
cfg = ald.Configuration.from_Poiseuille(rtp=rtp, uf=0.1, N=204800)
simulator = ald.Simulator(cfg)
dt = 1e-4
Nt = 1000000

# range to compute stats on configuration and print time.
stat_range = ald.InRange(start=0, stop=Nt, freq=10000)
# setup callbacks.
callbacks = [ald.StatsCallback(stat_range, "x"), ald.PrintCallback(stat_range)]

simulator.run(cfg, dt, Nt, callbacks=callbacks)
