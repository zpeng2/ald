import ald

rtp = ald.Pareto()
cfg = ald.Configuration.from_Poiseuille(rtp=rtp, uf=0.1, N=204800, dt=1e-4, Nt=1000000)
simulator = ald.Simulator(cfg)


# range to compute stats on configuration and print time.
stat_range = ald.InRange(start=0, stop=cfg.Nt, freq=50000)
# setup callbacks.
callbacks = [ald.MeanVariance(stat_range, "x", unwrap=True), ald.ETA(stat_range)]

simulator.run(cfg, callbacks=callbacks)
