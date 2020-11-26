from abc import abstractmethod, ABC
import pycuda.gpuarray as gpuarray
import pycuda.curandom
import pycuda.compiler as compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np

from .particles import AbstractParticle, AbstractRTP


class AbstractConfig:
    def __init__(self, particle, N):
        # particle needs to be a subtype of AbatractParticle
        if not isinstance(particle, AbstractParticle):
            raise TypeError("{} is not a subclass of AbstractParticle".format(particle))
        if not isinstance(N, int):
            raise TypeError("{} is not an integer.".format(N))
        # keep a copy of the particle object
        self.particle = particle
        # keep the number of particles
        self.N = N
        # common configuration data
        # initial configuration of the particle system
        self.x0 = gpuarray.GPUArray(N, dtype=np.float64)
        self.y0 = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta0 = gpuarray.GPUArray(N, dtype=np.float64)
        # config at the previous time step.
        self.x_old = gpuarray.GPUArray(N, dtype=np.float64)
        self.y_old = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta_old = gpuarray.GPUArray(N, dtype=np.float64)
        # config at the current time
        self.x = gpuarray.GPUArray(N, dtype=np.float64)
        self.y = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta = gpuarray.GPUArray(N, dtype=np.float64)
        # total number of boundary crossings in x and y
        self.passx = gpuarray.GPUArray(N, dtype=np.int32)
        self.passy = gpuarray.GPUArray(N, dtype=np.int32)
        # current time
        self.t = 0.0


class Config(AbstractConfig):
    def __init__(self, particle, N):
        super().__init__(particle, N)
        # additional configuration info specifically for RTPs.
        if isinstance(particle, AbstractRTP):
            # curandstate array.
            sizeof_state = pycuda.characterize.sizeof(
                "curandStateXORWOW", "#include <curand_kernel.h>"
            )
            self.state = cuda.mem_alloc(N * sizeof_state)
            # array for the current runtime drawed from its distribution
            self.tauR = gpuarray.GPUArray(N, dtype=np.float64)
            # time since last tumble.
            self.tau = gpuarray.GPUArray(N, dtype=np.float64)
