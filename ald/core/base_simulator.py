from abc import abstractmethod, ABC

from .configs import AbstractConfig

# TODO: in the template function for RTP, can add flow(basically arbitrary external velocities) easily, because for a simulation, we already know the flow parameters, they can be replaced at compile time by numbers, so that no additional arguments need to be passed to the general cuda kernel!!


class AbstractSimulator(ABC):
    """Langevin simulation of particles in 2D (channel or freespace.)"""

    @property
    @abstractmethod
    def particle(self):
        pass

    @property
    @abstractmethod
    def _isinitialized(self):
        # need to call self.initialize to initialize the simulator and system.
        pass

    @abstractmethod
    def initialize(self, cfg, *args, **kwargs):
        """Initialize the particle and simulation configurations."""
        pass

    @abstractmethod
    def update(self, cfg, dt, **kwargs):
        """One step of the Langevin simulation"""
        pass

    def run(self, cfg, dt, Nt, callbacks=None, **kwargs):
        """Run Langevin simulation"""
        if not self._isinitialized:
            self.initialize(cfg)

        for i in range(Nt):
            self.update(cfg, dt, **kwargs)
            if callbacks is not None:
                # accepts an iterable of callbacks.
                if not hasattr(callbacks, "__iter__"):
                    # maybe the user passed in a single callback function
                    # try wrap it with a list
                    callbacks = [callbacks]
                # run each callback function
                for callback in callbacks:
                    callback(i, cfg)

            cfg.t += dt
