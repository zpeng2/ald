from abc import abstractmethod, ABC
from ald.core.ic import InitialConfig
from ald.core.particle import AbstractParticle, AbstractRTP, Pareto, RTP
from ald.core.external_velocity import ExternalVelocity, EmptyVelocity, Poiseuille
from ald.core.boundary import AbstractDomain, Box
import pycuda.autoinit
import pycuda.curandom
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from jinja2 import Template
import os

# base cuda code.
cuda_code = """
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.141592653589793

extern "C" {
// initialize RNG
// passing in an array of states
// each thread has to have its own RNG
__global__ void initrand(curandState *__restrict__ state, const int N) {
  int seed = 0;
  int offset = 0;
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    curand_init(seed, tid, offset, &state[tid]);
  }
}

// generate Pareto run times.
// https://en.wikipedia.org/wiki/Pareto_distribution#Random_sample_generation
// taum is the minimum run time, alpha is the exponent
__device__ double pareto_runtime(curandState *state, double tauR,
                                 double alpha) {
  // !! tauR is the mean runtime, which is taum*alpha/(alpha-1)
  double taum = (alpha - 1.0) * tauR / alpha;
  // curand_uniform_double is uniformly distributed in (0,1)
  double U = curand_uniform_double(state);
  return taum / pow(U, 1.0 / alpha);
}

// draw run times for each active particle
__global__ void draw_pareto_runtimes(
    double *tauR, curandState *state, const int N, const double tauavg,
    const double alpha) { // for loop allows more particles than threads.
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    tauR[tid] = pareto_runtime(&state[tid], tauavg, alpha);
  }
}

// dummy function for constant runtime RTPs.
__device__ double constant_runtime(curandState *state, double tauR, ...) {
  return tauR;
}

} // extern C
"""


class AbstractCompiler(ABC):
    def __init__(
        self,
        particle=RTP(),
        domain=Box.from_freespace(),
        flow=EmptyVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(particle, AbstractParticle):
            raise TypeError()
        if not isinstance(domain, AbstractDomain):
            raise TypeError()
        if not isinstance(flow, ExternalVelocity):
            raise TypeError()
        self.particle = particle
        self.domain = domain
        self.flow = flow
        self.ic = ic
        # base cuda code that is used.
        self.cuda_code_base = cuda_code

    @abstractmethod
    def compile(self, *args, **kwargs):
        """Child class must define a concrete compile method."""
        pass

    @property
    @abstractmethod
    def cuda_code(self):
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


# class RTPCompiler(AbstractCompiler):
#     """Cuda code compiler for RTPs."""

#     def __init__(
#         self,
#         particle=RTP(),
#         box=Box.from_freespace(),
#         flow=EmptyVelocity(),
#         ic=InitialConfig(),
#     ):
#         if not isinstance(particle, AbstractRTP):
#             raise TypeError()
#         if not isinstance(box, AbstractDomain):
#             raise TypeError()
#         if not isinstance(flow, ExternalVelocity):
#             raise TypeError()
#         self.particle = particle
#         self.box = box
#         self.flow = flow
#         self.ic = ic
#         # cuda_code
#         self.cuda_code = cuda_code
#         # render rtp kernel template.
#         self._render_rtp_kernel()
#         # compile
#         self.compile()

#     def _render_rtp_kernel(self):
#         """Run template renderer"""
#         self.rtp_kernel_code = rtp_kernel_template.render(
#             runtime=self.particle.runtime_code,
#             ux=self.flow.ux,
#             uy=self.flow.uy,
#             omega=self.flow.omega,
#             bc=self.box.bc,
#         )

#     def log2file(self, log):
#         """Log cuda sourcecode to a cuda file for easy viewing"""
#         if not isinstance(log, str):
#             raise ValueError("invalid filename {}".format(log))
#         # log to a file
#         # if the filename already exist, append _new.cu
#         if os.path.isfile(log):
#             log += "new.cu"
#         with open(log, "w") as f:
#             f.write(self.cuda_code)
#         print("cuda source code saved at {}".format(log))
#         return None

#     def compile(self, log=None):
#         """Compile cuda source code"""
#         # combine cuda source codes
#         # make sure this function can be run multiple times
#         self.cuda_code = cuda_code
#         # append bd_rtp.
#         self.cuda_code += self.rtp_kernel_code
#         # append initial condition kernel
#         self.cuda_code += self.ic.cuda_code
#         if log is not None:
#             self.log2file(log)

#         module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
#         # get functions from cuda module
#         self.update_rtp = module.get_function("update_rtp")
#         self.initrand = module.get_function("initrand")
#         self.init_config = module.get_function("init_config")
#         self.draw_pareto_runtimes = module.get_function("draw_pareto_runtimes")
