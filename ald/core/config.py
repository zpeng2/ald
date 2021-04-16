from abc import abstractmethod, ABC
import pycuda.gpuarray as gpuarray
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
import h5py
from ald.core.particle import AbstractParticle, AbstractRTP


class AbstractConfig:
    def __init__(self, particle, domain, N, dt, Nt):
        # particle needs to be a subtype of AbatractParticle
        if not isinstance(particle, AbstractParticle):
            raise TypeError("{} is not a subclass of AbstractParticle".format(particle))
        if not isinstance(N, int):
            raise TypeError("{} is not an integer.".format(N))
        # keep a copy of the particle object
        self.particle = particle
        # keep a copy of the domain object
        self.domain = domain
        # keep the number of particles
        self.N = N
        # time step
        self.dt = dt
        # total number of steps
        self.Nt = Nt
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
        # curandstate array.
        sizeof_state = pycuda.characterize.sizeof(
            "curandStatePhilox4_32_10", "#include <curand_kernel.h>"
        )
        self.state = cuda.mem_alloc(N * sizeof_state)
        # current time
        self.t = 0.0

    def save2h5(self, file):
        """Save particle, domain simulation attributes to hdf5"""
        with h5py.File(file, "r+") as f:
            # write attributes only, not data arrays.
            # particle attributes.
            # save only scalar and numeric attributes.
            for attr, value in vars(self.particle).items():
                if np.isscalar(value) and np.isreal(value):
                    f.attrs[attr] = value
            # save domain info
            for attr, value in vars(self.domain).items():
                if np.isscalar(value) and np.isreal(value):
                    f.attrs[attr] = value
            # save simulation attributes
            f.attrs["dt"] = self.dt
            f.attrs["Nt"] = self.Nt
            f.attrs["N"] = self.N


class Config(AbstractConfig):
    def __init__(self, particle, domain, N, dt, Nt):
        super().__init__(particle, domain, N, dt, Nt)
        # additional configuration info specifically for RTPs.
        if isinstance(particle, AbstractRTP):
            # array for the current runtime drawed from its distribution
            self.tauR = gpuarray.GPUArray(N, dtype=np.float64)
            # time since last tumble.
            self.tau = gpuarray.GPUArray(N, dtype=np.float64)
