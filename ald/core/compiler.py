from abc import abstractmethod, ABC
from ald.core.kernel import AbstractKernel
from ald.core.config import AbstractConfig
from ald.core.ic import InitialConfig
from ald.core.particle import AbstractParticle, AbstractRTP, Pareto, RTP
from ald.core.external_velocity import ExternalVelocity, ZeroVelocity, Poiseuille
from ald.core.boundary import AbstractDomain, Box
import pycuda.autoinit
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from jinja2 import Template
import os

# base cuda code.
cuda_code_base = """
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define PI 3.141592653589793

extern "C" {
// initialize RNG
// passing in an array of states
// each thread has to have its own RNG
__global__ void initrand(curandStatePhilox4_32_10_t *state, const int N) {
  int seed = 1234;
  int offset = 0;
  /* Each thread gets same seed, a different sequence
    number, no offset */
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    curand_init(seed, tid, offset, &state[tid]);
  }
}

__device__ double uniform_rand(curandStatePhilox4_32_10_t *state, double a,
                                double b) {
return a+(b-a)*curand_uniform_double(state);
}

} // extern C
"""


class AbstractCompiler(ABC):
    def __init__(
        self,
        kernel,
        cfg,
        flow=ZeroVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(cfg, AbstractConfig):
            raise TypeError()
        if not isinstance(flow, ExternalVelocity):
            raise TypeError()
        if not isinstance(kernel, AbstractKernel):
            raise TypeError()
        self.kernel = kernel
        self.particle = cfg.particle
        self.domain = cfg.domain
        self.flow = flow
        self.ic = ic
        # base cuda code that is used.
        self.cuda_code_base = cuda_code_base
        # fully rendered cuda code.
        self.cuda_code = self.generate_cuda_code(cfg, flow)

    @abstractmethod
    def compile(self, *args, **kwargs):
        """Child class must define a concrete compile method."""
        pass

    @abstractmethod
    def generate_cuda_code(self, *args, **kwargs):
        """This should be the full cuda code"""
        pass

    def log2file(self, log):
        """Log cuda sourcecode to a cuda file for easy viewing"""
        if not isinstance(log, str):
            raise ValueError("invalid filename {}".format(log))
        # log to a file
        # if the filename already exist, append _new.cu
        if os.path.isfile(log):
            log += "new.cu"
        with open(log, "w") as f:
            f.write(self.cuda_code)
        print("cuda source code saved at {}".format(log))
        return None
