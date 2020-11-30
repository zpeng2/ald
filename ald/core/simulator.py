from abc import abstractmethod, ABC
from ald.core.compiler import AbstractCompiler
from ald.core.config import AbstractConfig
from ald.core.particle import RTP, Pareto
import numpy as np


class AbstractSimulator(ABC):
    """Langevin simulation of particles in 2D (channel or freespace.)"""

    def __init__(self, cfg, compiler, threadsPerBlock=None, nblocks=None):
        if not isinstance(cfg, AbstractConfig):
            raise TypeError()
        if not isinstance(compiler, AbstractCompiler):
            raise TypeError()
        # keep a copy of domain and particle
        self.domain = cfg.domain
        self.particle = cfg.particle

        # keep a copy of compiler
        self.compiler = compiler

        # this is set after calling self.initialize
        self.isinitialized = False

        # cuda launch parameters
        # cuda kernel launch parameter
        if threadsPerBlock is None:
            self.threadsPerBlock = 512
        else:
            self.threadsPerBlock = threadsPerBlock
        if nblocks is None:
            self.nblocks = cfg.N // self.threadsPerBlock + 1
            if self.nblocks > 500:
                self.nblocks = 500
        else:
            self.nblocks = nblocks

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize the particle and simulation configurations."""
        self.isinitialized = True

    @abstractmethod
    def update(self, *args, **kwargs):
        """One step of the Langevin simulation"""
        pass

    def launch_kernel(self, func, *args):
        """Launch cuda kernel func"""
        func(
            *args,
            block=(self.threadsPerBlock, 1, 1),
            grid=(self.nblocks, 1),
        )
        return None

    def run(self, cfg, callbacks=None):
        """Run Langevin simulation"""
        if not self.isinitialized:
            self.initialize(cfg)

        for i in range(cfg.Nt + 1):
            self.update(cfg)
            cfg.t = cfg.dt * (i + 1)

            if callbacks is not None:
                # accepts an iterable of callbacks.
                if not hasattr(callbacks, "__iter__"):
                    # maybe the user passed in a single callback function
                    # try wrap it with a list
                    callbacks = [callbacks]
                # run each callback function
                for callback in callbacks:
                    callback(i, cfg)


class Simulator(AbstractSimulator):
    """Langevin simulation of RTPs in 2D (channel or freespace.)"""

    def __init__(self, cfg, compiler, threadsPerBlock=None, nblocks=None):
        # not initialized yet
        super().__init__(
            cfg, compiler, threadsPerBlock=threadsPerBlock, nblocks=nblocks
        )

    def initialize(self, cfg):
        """Initialize the particle and simulation configurations."""
        # initialize RNG
        self.launch_kernel(self.compiler.initrand, cfg.state, np.int32(cfg.N))
        # initialize particle configuration
        self.launch_kernel(
            self.compiler.init_config,
            cfg.state,
            cfg.x0,
            cfg.y0,
            cfg.theta0,
            np.int32(cfg.N),
        )

        # need to fill x,y,theta old too
        cfg.x_old = cfg.x0.copy()
        cfg.y_old = cfg.y0.copy()
        cfg.theta_old = cfg.theta0.copy()
        # need to initialize passx and passy to 0
        cfg.passx.fill(0)
        cfg.passy.fill(0)
        # similarly, time since last reorientation is zero,.
        cfg.tau.fill(0.0)
        # need to initialize the runtime for each particle.
        if isinstance(self.particle, RTP):
            # constant runtime.
            cfg.tauR.fill(self.particle.tauR)
        elif isinstance(self.particle, Pareto):
            # Pareto distributed runtimes.
            self.launch_kernel(
                self.compiler.draw_pareto_runtimes,
                cfg.tauR,
                cfg.state,
                np.int32(cfg.N),
                np.float64(self.particle.tauR),
                np.float64(self.particle.alpha),
            )
        else:
            raise NotImplementedError()
        # indicate that the system is initialized
        super().initialize()

        return None

    def update(self, cfg):
        """One step of the Langevin simulation."""
        self.launch_kernel(
            self.compiler.update_rtp,
            cfg.x_old,
            cfg.y_old,
            cfg.theta_old,
            cfg.x,
            cfg.y,
            cfg.theta,
            cfg.passx,
            cfg.passy,
            cfg.state,
            cfg.tauR,
            cfg.tau,
            np.float64(cfg.particle.U0),
            np.float64(cfg.dt),
            np.int32(cfg.N),
        )
