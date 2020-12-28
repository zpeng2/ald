import ald
import numpy as np
import h5py

gamma = 1
Pes = 1
Pe = 1
params = ald.RheologyParams(gamma=gamma, Pes=Pes, Pe=Pe)
file = "gamma{:.3f}Pes{:.3f}Pe{:.3f}.h5".format(params.gamma, params.Pes, params.Pe)
with h5py.File(file, "w") as _:
    pass

cfg = ald.RheologyConfig(params)
cfg.save2h5(file)

domain = cfg.domain

ic = ald.InitialConfig(
    x=ald.Uniform(domain.left, domain.right),
    y=ald.Uniform(domain.bottom, domain.top),
    theta=ald.Uniform(0, 2 * np.pi),
)

kernel = ald.RheologyKernel(params)

flow = ald.ZeroVelocity()

compiler = ald.ABPCompiler(kernel, cfg, flow, ic)
#  compiler.log2file("rheo.cu")
compiler.compile()


simulator = ald.ABPSimulator(cfg, compiler)

# setup callbacks.


#  configsaver = ald.ConfigSaver(runner, file, variables=["x"], unwrap=[False])
eta = ald.ETA(ald.RangedRunner(start=0, stop=cfg.Nt, freq=20000))

runner = ald.RangedRunner(start=int(params.Nt / 2), stop=params.Nt, freq=100)
runner = ald.RangedRunner(start=int(params.Nt  /  2), stop=params.Nt, freq=100)
dx = ald.SimpleMean(runner, "dx", keep_time=True)
dy = ald.SimpleMean(runner, "dy", keep_time=True)
# # callbacks = [eta, configsaver, force]
callbacks = [eta, dx, dy]


simulator.run(cfg, callbacks=callbacks)
# save further data
#   force.save2h5(file, "dx")
# save other attributes
dx.save2h5(file, "dx")
dy.save2h5(file, "dy")


# if __name__ == "__main__":
#     for u in U0:
#         simulate(u)
