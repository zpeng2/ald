# Pareto particles confined between two parallel plates.
import ald
import numpy as np

# particle parameter
U0 = 1.0
# tauR is the mean runtime!
tauR = 1.0
alpha = 1.2

particle = ald.Pareto(U0=U0, tauR=tauR, alpha=alpha)

# no flow
flow = ald.ZeroVelocity()

# channel width
H = 1.0

domain = ald.Box.from_channel(Ly=H)

# initial condition
ic = ald.InitialConfig(
    x=ald.Point((domain.left + domain.right) / 2),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)


# number of particles
N = 204800
# time step
dt = 1e-4
# total steps
Nt = 2000000

# setup system configuration
cfg = ald.Config(particle, domain, N=N, dt=dt, Nt=Nt)


compiler = ald.RTPCompiler(cfg, flow, ic)

simulator = ald.RTPSimulator(cfg, compiler)


# setup callbacks.
file = "U{}tauR{}.h5".format(U0, tauR)
# save configuraton.
# callback controller
runner = ald.RangedRunner(start=0, stop=cfg.Nt, freq=10000)
configsaver = ald.ConfigSaver(runner, file, variables=["x", "y", "theta"])
# print out ETA
eta = ald.ETA(ald.RangedRunner(start=0, stop=cfg.Nt, freq=20000))

# compute mean and variance of displacement along the channel
x = ald.DisplacementMeanVariance(
    ald.RangedRunner(start=0, stop=cfg.Nt, freq=20000), "x", unwrap=True
)
# y = ald.MeanVariance(runner, "y", unwrap=True)
callbacks = [eta, configsaver, x]

# run the simulation
simulator.run(cfg, callbacks=callbacks)

# save other information.
# save particle, domain and simulation attributes.
cfg.save2h5(file)
# save mean variance of x
x.save2h5(file, "x")
