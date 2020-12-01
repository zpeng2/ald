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
