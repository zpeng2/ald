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


class Always(CallbackRunner):
    def iscomputing(self, i):
        return True


class RangedRunner(CallbackRunner):
    """Run callback if an index is in a range."""

    def __init__(self, start, stop, freq):
        # make a range that includes the stop as a point
        # number of groups that have freq numbers
        k = (stop - start + 1) // freq
        start = stop - (k - 1) * freq
        # python range is not right-end inclusive
        stop = stop + 1
        self._range = range(start, stop, freq)

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


class DisplacementMeanVariance(Callback):
    """Compute mean and variance of a GPUArray."""

    def __init__(self, runner, variable, unwrap=False):
        super().__init__(runner)
        # which cfg attribute to do statistics.
        if not variable in ["x", "y"]:
            raise ValueError("invalid variable: {}".format(variable))
        self.variable = variable
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
        x = getattr(cfg, self.variable).copy()
        # initial location
        x0 = getattr(cfg, self.variable + "0").copy()
        # unwrap
        if self.unwrap:
            # get the boundary crossing array.
            passx = getattr(cfg, "pass" + self.variable).copy()
            # do unwrap
            L = getattr(cfg.domain, "L" + self.variable)
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

    def save2h5(self, file, group):
        """Save m and t to file"""
        # save
        with h5py.File(file, "r+") as f:
            f[os.path.join(group, "t")] = self.t
            f[os.path.join(group, "m")] = self.m
            f[os.path.join(group, "v")] = self.v


class SimpleMean(Callback):
    """Compute simple mean"""

    def __init__(self, runner, variable, keep_time=False):
        super().__init__(runner)
        self.variable = variable
        # instantiate arrays to store mean and variance
        self.m = np.zeros(len(self.runner))
        # keep track of time
        self.keep_time = keep_time
        if keep_time:
            self.t = np.zeros_like(self.m)
        # index used to store values
        self.idx = 0

    def compute_mean(self, cfg):
        # get data.
        x = getattr(cfg, self.variable).copy()
        N = len(x)
        mean_x = gpuarray.sum(x) / N
        # copy to cpu and flatten
        mean_x = float(mean_x.get())
        return mean_x

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            self.m[self.idx] = self.compute_mean(cfg)
            if self.keep_time:
                self.t[self.idx] = cfg.t
            self.idx += 1
        return None

    def save2h5(self, file, group):
        """Save m and t to file"""
        # save
        with h5py.File(file, "r+") as f:
            f[os.path.join(group, "t")] = self.t
            f[os.path.join(group, "m")] = self.m


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
    def __init__(
        self, runner, file, variables=["x", "y", "theta"], unwrap=[False, False, False]
    ):
        super().__init__(runner)
        self.variables = variables
        if not isinstance(file, str):
            raise TypeError("invalid filename")
        self.file = file
        # length of vcariables and unwrap should be the same
        if len(variables) != len(unwrap):
            raise ValueError("lengths of variables and unwrap do not match")
        # theta cannot unwrap
        if "theta" in variables:
            i = variables.index("theta")
            if unwrap[i]:
                raise ValueError("cannot unwrap theta")
        self.unwrap = unwrap
        # keep a frame counter
        self.counter = 0
        # if file exists, error
        if os.path.exists(self.file):
            raise ValueError("file: {} exists.".format(self.file))
        else:
            # create file
            with open(self.file, "w") as f:
                pass

    def get_config(self, variable, cfg, unwrap):
        # need to unwrap if requested
        # need to copy data! because it will be modified later.
        variable_gpu = getattr(cfg, variable).copy()
        if unwrap:
            variable_gpu += getattr(cfg, "pass" + variable).copy()
        # need to copy to cpu
        return variable_gpu.get()

    def __call__(self, i, cfg):
        if self.runner.iscomputing(i):
            with h5py.File(self.file, "r+") as f:
                path = "config/{}/".format(self.counter)
                # keep track of time
                f[path + "t"] = cfg.t
                # save configuration
                for variable, unwrap in zip(self.variables, self.unwrap):
                    configpath = path + "{}".format(variable)
                    f[configpath] = self.get_config(variable, cfg, unwrap)

            # need to update counter
            self.counter += 1


# class RuntimeSaver(Callback):
#     """Save the sampled runtime history of aprticles."""

#     def __init__(self, file):
#         runner = Always()
#         super().__init__(runner)
#         if not isinstance(file, str):
#             raise TypeError("invalid filename")
#         self.file = file
#         # array of runtime history regardless of particle id.
#         self.runtimes = []

#     def __call__(self, i, cfg):
#         if self.runner.iscomputing(i):
#             tauR = cfg.tauR.get()
#             tau = cfg.tau.get()
#             # if tau is zero, means I just tumbled and my runtime is newly generated.
#             # and I need to record it.
#             for i in range(cfg.N):
#                 if tau[i] == 0.0:
#                     self.runtimes.append(tauR[i])

#     def save2h5(self):
#         with h5py.File(self.file, "r+") as f:
#             path = "runtimes"
#             # keep track of time
#             f[path] = np.array(self.runtimes)
