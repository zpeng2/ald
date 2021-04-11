import ald
import numpy as np
import h5py

U0 = 5.0
tauR = 1
alpha = 1.2

# specify RTP type
particle = ald.Lomax(U0=U0, tauR=tauR, alpha=alpha)

H = 1.0
uf = 0.1
flow = ald.Poiseuille(uf=uf, H=H)

domain = ald.Box(bottom=-H / 2, top=H / 2)


ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)


cfg = ald.Config(particle, domain, N=100_000, dt=1e-4, Nt=4000000)

# Specify the channel problem
kernel = ald.RTPChannelKernel()

compiler = ald.RTPCompiler(kernel, cfg, flow, ic)

compiler.compile()

simulator = ald.RTPSimulator(cfg, compiler)

file = "U{:.3f}tauR{:.3f}.h5".format(U0, tauR)
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
