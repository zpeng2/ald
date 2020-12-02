from numpy.lib.arraysetops import isin
import pycuda.gpuarray as gpuarray
from abc import abstractmethod, ABC
import numpy as np
import time
import datetime
import h5py
import os


class CallbackRunner(ABC):
    """Decide whether to run a callback."""

    @abstractmethod
    def iscomputing(self, i):
        pass


class RangedRunner(CallbackRunner):
    """Run callback if an index is in a range."""

    def __init__(self, start, stop, freq):
        # python range is end-exclusive.
        self._range = range(start, stop + freq, freq)

    @classmethod
    def from_forward_count(cls, start=0, freq=1, count=10):
        """Constructor that takes the start, freq and number of points."""
        stop = start + (count - 1) * freq
        return cls(start, stop, freq)

    @classmethod
    def from_backward_count(cls, stop=100, freq=1, count=10):
        """Constructor that takes the stop, freq and number of points."""
        start = stop - (count - 1) * freq
        # require start to be positive, otherwise pointless
        if start < 0:
            raise ValueError("start={} is negative".format(start))
        return cls(start, stop, freq)

    def iscomputing(self, i):
        if i in self._range:
            return True
        else:
            return False

    def __len__(self):
        return len(self._range)


class Callback(ABC):
    """Callback that computes/stores information as the simulation evolves."""

    def __init__(self, runner):
        if not isinstance(runner, CallbackRunner):
            raise TypeError()
        self.runner = runner

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MeanVariance(Callback):
    """Compute mean and variance of a GPUArray."""

    def __init__(self, runner, attr, unwrap=False):
        super().__init__(runner)
        # which cfg attribute to do statistics.
        self.attr = attr
        # unwrap periodic to get absolute positions.
        self.unwrap = unwrap
        # instantiate arrays to store mean and variance
        self.m = np.zeros(len(self.runner))
        self.v = np.zeros_like(self.m)
        # keep track of time
        self.t = np.zeros_like(self.m)
        # index used to store values
        self.idx = 0

    def mean_variance(self, cfg):
        # get data.
        x = getattr(cfg, self.attr)
        # initial location
        x0 = getattr(cfg, self.attr + "0")
        # unwrap
        if self.unwrap:
            # get the boundary crossing array.
            passx = getattr(cfg, "pass" + self.attr)
            # do unwrap
            L = getattr(cfg.domain, "L" + self.attr)
            # need the relative to the initial positions
            x -= x0
            x += passx * L
        else:
            x -= x0
        N = len(x)
        mean_x = gpuarray.sum(x) / N
        # copy to cpu and flatten
        mean_x = float(mean_x.get())
        # compute variance
        variance_x = gpuarray.sum((x - mean_x) ** 2) / N
        variance_x = float(variance_x.get())
        return mean_x, variance_x

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            m, v = self.mean_variance(cfg)
            self.m[self.idx] = m
            self.v[self.idx] = v
            self.t[self.idx] = cfg.t
            self.idx += 1
        return None


class PrintCallback(Callback):
    def __init__(self, runner):
        super().__init__(runner)

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            print("t = {:.3f}".format(cfg.t))


class ETA(Callback):
    def __init__(self, runner):
        super().__init__(runner)

    def __call__(self, i, cfg):
        """Printout TPS and ETA."""
        if i == 0:
            self.start = time.time()
        else:
            if self.runner.iscomputing(i):
                elapsed = time.time() - self.start
                # timesteps per second
                tps = int(i / elapsed)
                # estimated remaining time
                # total estimated time
                total = cfg.Nt / tps
                eta = total - elapsed
                # convert seconds to human friendly format hh:mm:ss
                eta_human = str(datetime.timedelta(seconds=int(eta)))
                # print output
                print("TPS:{0}, ETA:{1}".format(tps, eta_human))


class ConfigSaver(Callback):
    def __init__(self, runner, file, variables=["x", "y", "theta"]):
        super().__init__(runner)
        self.variables = variables
        if not isinstance(file, str):
            raise TypeError("invalid filename")
        self.file = file
        # keep a frame counter
        self.counter = 0
        # if file doesnt exist
        if not os.path.exists(self.file):
            # create file
            with open(self.file, "w") as f:
                pass

    def get_config(self, variable, cfg):
        variable_gpu = getattr(cfg, variable)
        return variable_gpu.get()

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            with h5py.File(self.file, "r+") as f:
                for variable in self.variables:
                    path = "config/{}/{}".format(self.counter, variable)
                    f[path] = self.get_config(variable, cfg)
                    # need to update counter
                    self.counter += 1
