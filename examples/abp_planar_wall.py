import ald
import numpy as np


# parameters
U0 = 1
DT = 1.0
tauR = 1.0

particle = ald.ABP(U0=U0, DT=DT, tauR=tauR)

flow = ald.ZeroVelocity()

domain = ald.Box(left=0, right=10, bottom=-0.5, top=0.5)

ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)

# simulation parameters
dt = 1e-4
Nt = 2000000

cfg = ald.Config(particle, domain, N=204800, dt=dt, Nt=Nt)

kernel = ald.PlanarWallKernel(cfg)


compiler = ald.ABPCompiler(kernel, cfg, flow, ic)

simulator = ald.ABPSimulator(cfg, compiler)


runner = ald.RangedRunner(start=Nt//2, stop=cfg.Nt, freq=10000)
# setup callbacks.
file = "U{}DT{}DR{}.h5".format(U0, DT, particle.DR)
configsaver = ald.ConfigSaver(runner, file, variables=["x"])
eta = ald.ETA(ald.RangedRunner(start = 0, stop=cfg.Nt, freq=20000))
force = ald.SimpleMean(runner, "dx", keep_time=True)

callbacks = [eta, configsaver, force]


simulator.run(cfg, callbacks=callbacks)
# save further data
force.save2h5(file, "dx")
# save other attributes
cfg.save2h5(file)
