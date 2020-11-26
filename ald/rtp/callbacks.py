from ald.rtp.rtp import Simulator
import pycuda.gpuarray as gpuarray
from abc import abstractmethod, ABC
import numpy as np


class InRange:
    """Validate whether an integer is in a python range."""

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

    def __call__(self, i):
        """Check if a given integer is included in _range."""
        if i in self._range:
            return True
        else:
            return False

    def __len__(self):
        return len(self._range)


class Callback(ABC):
    """Callback that computes/stores information as the simulation evolves."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class StatsCallback(Callback):
    """Compute mean and variance of a GPUArray."""

    def __init__(self, inrange, attr):
        if not isinstance(inrange, InRange):
            raise TypeError("wrong type {}".format(type(inrange)))
        self.iscomputing = inrange
        # which cfg attribute to do statistics.
        self.attr = attr
        # instantiate arrays to store mean and variance
        self._mean = np.zeros(len(inrange))
        self._variance = np.zeros_like(self._mean)
        # counter for storing data into mean and variance arrays.
        self.i = 0

    def mean_variance(self, cfg):
        # get data.
        x = getattr(cfg, self.attr)
        N = len(x)
        mean_x = gpuarray.sum(x) / N
        # copy to cpu and flatten
        mean_x = float(mean_x.get())
        # compute variance
        variance_x = gpuarray.sum((x - mean_x ** 2)) / N
        variance_x = float(variance_x.get())
        return mean_x, variance_x

    def __call__(self, i, cfg):
        if self.iscomputing(i):
            m, v = self.mean_variance(cfg)
            self._mean[self.i] = m
            self._variance[self.i] = v
            self.i += 1
        return None


class PrintCallback(Callback):
    def __init__(self, inrange):
        if not isinstance(inrange, InRange):
            raise TypeError("wrong type {}".format(type(inrange)))
        self.iscomputing = inrange

    def __call__(self, i, cfg):
        if self.iscomputing(i):
            print("t = {:.3f}".format(cfg.t))
