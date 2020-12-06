import ald
import numpy as np

# only one dimensionless group l/delta, so only change U0.
# parameters
U0 = np.logspace(-1, 1, 7)


def simulate(U0):
    DT = 1.0
    tauR = 1.0
    particle = ald.ABP(U0=U0, DT=DT, tauR=tauR)

    flow = ald.ZeroVelocity()
    # need the rhs to be much large than run length
    L = U0 * tauR * 10
    domain = ald.Box(left=0, right=L, bottom=-0.5, top=0.5)

    ic = ald.InitialConfig(
        x=ald.Uniform(domain.left, domain.right),
        y=ald.Uniform(domain.bottom, domain.top),
        theta=ald.Uniform(0, 2 * np.pi),
    )

    # simulation parameters
    dt = 1e-4
    Nt = 4000000

    cfg = ald.Config(particle, domain, N=204800, dt=dt, Nt=Nt)

    kernel = ald.ABPSingleWallKernel()

    compiler = ald.ABPCompiler(kernel, cfg, flow, ic)
    compiler.compile()

    simulator = ald.ABPSimulator(cfg, compiler)

    # setup callbacks.
    file = "U{:.3f}singlewall.h5".format(U0)
    runner = ald.RangedRunner.from_backward_count(stop=cfg.Nt, freq=10000, count=100)
    configsaver = ald.ConfigSaver(runner, file, variables=["x"], unwrap=[False])
    eta = ald.ETA(ald.RangedRunner(start=0, stop=cfg.Nt, freq=20000))
    force = ald.SimpleMean(runner, "dx", keep_time=True)

    # callbacks = [eta, configsaver, force]
    callbacks = [eta, configsaver, force]

    simulator.run(cfg, callbacks=callbacks)
    # save further data
    force.save2h5(file, "dx")
    # save other attributes
    cfg.save2h5(file)


if __name__ == "__main__":
    for u in U0:
        simulate(u)
