# Pareto particles confined between two parallel plates.
import ald
import numpy as np
import h5py
import os


def simulate(lH=1, Peg=1.0, alpha=1.2):
    # Convert ell/H = U_0tauR/H  and
    #  Peg = 2u_f\tau_R/H to dimensional values.
    tauR = 1.0
    H = 1.0
    U0 = lH * H / tauR
    uf = 0.5 * H * Peg / tauR

    # note that tauR is the averahe runtime.
    particle = ald.Lomax(U0=U0, tauR=tauR, alpha=alpha)

    # Poiseuille flow
    flow = ald.Poiseuille(uf=uf, H=H)
    # simulation box
    domain = ald.Box(bottom=-H / 2, top=H / 2, left=-H / 2, right=H / 2)

    # initial condition
    ic = ald.InitialConfig(
        x=ald.Point((domain.left + domain.right) / 2),
        y=ald.Uniform(domain.bottom, domain.top),
        theta=ald.Uniform(0, 2 * np.pi),
    )

    # number of particles
    N = 300_000
    # determine timescales in our problem
    # flow timescale
    timescales = []
    if uf != 0.0:
        timescales.append(H / uf)
    if U0 != 0.0:
        timescales.append(H / U0)
    timescales.append(tauR)
    tmin = np.min(timescales)
    tmax = np.max(timescales)

    # time step
    dt = 1e-4 * tmin
    # final time
    tf = 1000 * tmax
    # total steps
    Nt = int(tf / dt)

    # setup system configuration
    cfg = ald.Config(particle, domain, N=N, dt=dt, Nt=Nt)

    kernel = ald.RTPChannelKernel()

    compiler = ald.RTPCompiler(kernel, cfg, flow, ic)
    compiler.compile()

    simulator = ald.RTPSimulator(cfg, compiler)

    # setup callbacks.
    file = "lH{:.3f}Peg{:.3f}alpha{:.3f}.h5".format(lH, Peg, alpha)

    # save configuraton.
    # callback controller
    # x is periodic, unwrap to save the absolute position
    configsaver = ald.ConfigSaver(
        ald.RangedRunner.from_backward_count(stop=cfg.Nt, freq=1000, count=200),
        file,
        variables=["x", "y", "theta"],
        unwrap=[True, False, False],
    )

    # print out ETA
    periodic = ald.RangedRunner(start=0, stop=cfg.Nt, freq=20000)
    eta = ald.ETA(periodic)

    # compute mean and variance of displacement along the channel
    x = ald.DisplacementMeanVariance(periodic, "x", unwrap=True)
    # record wall fraction
    wallfraction = WallFraction(periodic)

    # y = ald.MeanVariance(runner, "y", unwrap=True)
    callbacks = [eta, configsaver, x, wallfraction]
    # callbacks = [eta, configsaver, x]

    # run the simulation
    simulator.run(cfg, callbacks=callbacks)

    # save other information.
    # save particle, domain and simulation infos.
    cfg.save2h5(file)
    # save mean variance of x
    x.save2h5(file, "x")
    wallfraction.save2h5(file, "wallfraction")


class WallFraction(ald.Callback):
    """Compute fraction of particles that are on both walls."""

    def __init__(self, runner, keep_time=True):
        super().__init__(runner)
        # instantiate arrays to store fraction
        self.f = np.zeros(len(self.runner))
        # keep track of time
        self.keep_time = keep_time
        if keep_time:
            self.t = np.zeros_like(self.f)
        # index used to store values
        self.idx = 0

    def compute_fraction(self, cfg):
        # get data.
        y = getattr(cfg, "y")
        # copy to cpu
        y = y.get()
        Nw = np.sum([cfg.domain.bottom < val < cfg.domain.top for val in y])
        fw = Nw / cfg.N
        return fw

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            self.f[self.idx] = self.compute_fraction(cfg)
            if self.keep_time:
                self.t[self.idx] = cfg.t
            self.idx += 1
        return None

    def save2h5(self, file, group):
        """Save m and t to file"""
        # save
        with h5py.File(file, "r+") as f:
            f[os.path.join(group, "t")] = self.t
            f[os.path.join(group, "f")] = self.f


if __name__ == "__main__":
    lH = [0.01, 0.1, 1]
    Peg = [0.01, 0.1, 1, 10]
    for peg in Peg:
        simulate(lH=1.0, Peg=peg, alpha=1.2)
