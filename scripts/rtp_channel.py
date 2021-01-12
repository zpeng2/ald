import ald
import numpy as np
import h5py
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda


U0 = 1.0
tauR = 0.3
# specify RTP type
particle = ald.ExponentialRTP(U0=U0, tauR=tauR)

flow = ald.ZeroVelocity()

H = 1.0
domain = ald.Box(bottom=-H / 2, top=H / 2)


ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)


cfg = ald.Config(particle, domain, N=204800, dt=1e-4, Nt=4_000_000)

# Specify the channel problem
kernel = ald.RTPChannelKernel()

compiler = ald.RTPCompiler(kernel, cfg, flow, ic)
#compiler.log2file("a.cu")
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

class DebugCallback(ald.Callback):
    def __init__(self, runner):
        super().__init__(runner)

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            y = cfg.y.get()
            y = y.min()
            #y = float(y.get())
            if y < -0.5:
                print(y)

debug = DebugCallback(ald.RangedRunner(start=0, stop=cfg.Nt, freq=1))



callbacks = [x, y, eta, configsaver]#, debug]

simulator.run(cfg, callbacks=callbacks)

# save particle, domain and simulation attributes.
cfg.save2h5(file)
# save mean variance of x
x.save2h5(file, "x")
y.save2h5(file, "y")
