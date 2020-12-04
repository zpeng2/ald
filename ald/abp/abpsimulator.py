from ald.core.simulator import AbstractSimulator
import numpy as np
from ald.core.particle import ABP


class ABPSimulator(AbstractSimulator):
    """Langevin simulation of ABP in 2D (channel or freespace.)"""

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

        # indicate that the system is initialized
        super().initialize()

        return None

    def update(self, cfg):
        """One step of the Langevin simulation."""
        self.compiler.kernel.update(
            self.compiler.update, cfg, self.threadsPerBlock, self.nblocks
        )
