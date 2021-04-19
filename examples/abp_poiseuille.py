import ald
import numpy as np
import h5py


Pe = 1
lH = 10
ld = 30
# Set scales to compute other parameters.
DT=1.0
H = 1.0
uf = Pe*DT/H
ell = lH*H
delta = ell/ld
tauR  =delta**2/DT
U0 = ell/tauR

# specify RTP type
particle = ald.ABP(U0=U0, tauR=tauR, DT=DT)

flow = ald.Poiseuille(uf=uf, H=H)


domain = ald.Box(bottom=-H / 2, top=H / 2, left=-H / 2, right=H / 2)


ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)


# determine timescales in our problem
# flow timescale
timescales = []
if uf != 0.0:
    timescales.append(H / uf)
if U0 != 0.0:
    timescales.append(H / U0)
timescales.append(tauR)
if DT != 0.0:
    timescales.append(H**2/DT)
tmin = np.min(timescales)
tmax = np.max(timescales)

# time step
dt = 1e-3 * tmin
# final time
tf = 1000 * tmax
# total steps
Nt = int(tf / dt) +1

# number of ABPs,
N = 200_000

cfg = ald.Config(particle, domain, N=N, dt=dt, Nt=Nt)

# Specify the channel problem
kernel = ald.ABPChannelKernel()

compiler = ald.ABPCompiler(kernel, cfg, flow, ic)

compiler.compile()

simulator = ald.ABPSimulator(cfg, compiler)

file = "ABPU{:.3f}tauR{:.3f}DT{:.3f}.h5".format(U0, tauR, DT)
# create an empty file
# with h5py.File(file, "w") as f:
#     pass
# range to compute stats on configuration and print time.
runner = ald.RangedRunner(start=0, stop=cfg.Nt, freq=5000)
# setup callbacks.
x = ald.DisplacementMeanVariance(runner, "x", unwrap=True)
y = ald.DisplacementMeanVariance(runner, "y", unwrap=False)
eta = ald.ETA(ald.RangedRunner(start=0, stop=cfg.Nt, freq=50000))

# save configuraton.
# callback controller
# x is periodic, unwrap to save the absolute position
configsaver = ald.ConfigSaver(
    ald.RangedRunner.from_backward_count(stop=cfg.Nt, freq=10000, count=100),
    file,
    variables=["x", "y", "theta"],
    unwrap=[True, False, False],
)


callbacks = [x, y, eta, configsaver]

simulator.run(cfg, callbacks=callbacks)

# save particle, domain and simulation attributes.
cfg.save2h5(file)
# save mean variance of x
x.save2h5(file, "x")
y.save2h5(file, "y")
