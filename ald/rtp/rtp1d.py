# A 1D model with no-flux walls at ends.
# in this case, the orientation becomes +-1
from ald.rtp.rtpcompiler import AbstractCompiler
from jinja2 import Template
from ald.rtp.rtpkernels import AbstractRTPKernel
import pycuda.gpuarray as gpuarray
import numpy as np
from ald.rtp.rtpsimulator import RTPSimulator
import pycuda.curandom as curand
from ald.core.external_velocity import ZeroVelocity
from ald.core.ic import InitialConfig
from ald.core.particle import AbstractRTP
import pycuda.compiler as compiler
import pycuda


template1DRTP = Template(
    """
extern "C" {
__global__ void draw_runtimes(double *tauR,
    curandState *state,
    const int N)
{ // for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
    tid += blockDim.x * gridDim.x) {
    tauR[tid] = {{runtime}};
    }
}

// draw +1/-1 uniformly.
__global__ void draw_binary(int *arr,
    curandState *state,
    const int N)
{ // for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
    tid += blockDim.x * gridDim.x)
    {
        double U = curand_uniform_double(&state[tid]);
        if (U <= 0.5)
        {
            arr[tid] = -1;
        }
        else {arr[tid] = 1;}
    }
}



// evolution of RTPs in 1D.
__global__ void
update(double *__restrict__ x,          // position x
       int *__restrict__ direction,      // +1 or -1
       curandState *__restrict__ state, // RNG state
       double *__restrict__ tauR, // reorientation time for each active particle
       double *__restrict__ tau,  // time since last tumble
       double U0,                 // swim speed
       double L,    // simulation box length in x
       double dt,   // time step
       int N)

{
  // for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
      tid += blockDim.x * gridDim.x) {
  // need to tumble
if (tau[tid] >= tauR[tid]) {
  // tumbles between +1 and -1: randomly
//  double U = curand_uniform_double(&state[tid]);
//  // only 50% chance of tumbling away
//  if (U <= 0.5) {direction[tid] *= -1;}
  direction[tid] *= -1; // always changing direction once runtime is reached.
  // reset time since last tumble to zero.
  tau[tid] = 0.0;
  // after tumbling, need to draw a new tumbling time.
  tauR[tid] = {{runtime}};
}
// next update the position
x[tid] += dt * U0 * direction[tid];

// need to update time since last tumble.
tau[tid] += dt;

// x in [-L/2,L/2]
if (x[tid] < -L / 2.0) {
  x[tid] = -L/2.0;
} else if (x[tid] > L / 2.0) {
  x[tid] = L/2.0;
}
} // end thread loop.
} // end kernel.
}
"""
)


class Confined1DRTPKernel(AbstractRTPKernel):
    """1D two walls."""

    def __init__(self):
        # no additional args
        arg_list = ""
        super().__init__(arg_list)

    def generate_cuda_code(self, cfg, *args, **kwargs):
        """This kernel is hard coded"""
        # need to add a new container for the orientation
        setattr(
            cfg,
            "direction",
            gpuarray.GPUArray(cfg.N, np.int32),
        )
        kernel = template1DRTP.render(runtime=cfg.particle.runtime_code)
        return kernel

    def update(self, func, cfg, threads, blocks):
        func(
            cfg.x,
            cfg.direction,
            cfg.state,
            cfg.tauR,
            cfg.tau,
            np.float64(cfg.particle.U0),
            np.float64(cfg.domain.Lx),
            np.float64(cfg.dt),
            np.int32(cfg.N),
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )

        return None


class RTP1DCompiler(AbstractCompiler):
    """Cuda code compiler for RTPs."""

    def __init__(
        self,
        kernel,
        cfg,
        flow=ZeroVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(cfg.particle, AbstractRTP):
            raise TypeError()
        if not isinstance(kernel, AbstractRTPKernel):
            raise TypeError()

        super().__init__(kernel, cfg, flow=flow, ic=ic)

    def generate_cuda_code(self, cfg, flow):
        # combine cuda source codes
        # make sure this function can be run multiple times
        # do not touch self.cuda_code_base.
        # get the base code
        code = self.cuda_code_base
        # runtime generation kernel
        code += self.particle.runtime_device_code
        # append bd_code.
        code += self.kernel.generate_cuda_code(cfg, flow)
        # append initial condition kernel
        code += self.ic.cuda_code
        return code

    def compile(self, log=None):
        """Compile cuda source code"""
        module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
        # get functions from cuda module
        self.update = module.get_function("update")
        self.initrand = module.get_function("initrand")
        self.init_config = module.get_function("init_config")
        self.draw_runtimes = module.get_function("draw_runtimes")
        self.draw_binary = module.get_function("draw_binary")


class RTP1DSimulator(RTPSimulator):
    """Langevin simulation of RTPs in 2D (channel or freespace.)"""

    def __init__(self, cfg, compiler, threadsPerBlock=None, nblocks=None):
        # not initialized yet
        super().__init__(
            cfg, compiler, threadsPerBlock=threadsPerBlock, nblocks=nblocks
        )

    def initialize(self, cfg):
        """Initialize the particle and simulation configurations."""
        # initialize directiona rray
        super().initialize(cfg)
        self.compiler.draw_binary(
            cfg.direction,
            cfg.state,
            np.int32(cfg.N),
            block=(self.threadsPerBlock, 1, 1),
            grid=(self.nblocks, 1),
        )

        return None

    def update(self, cfg):
        """One step of the Langevin simulation."""
        self.compiler.kernel.update(
            self.compiler.update, cfg, self.threadsPerBlock, self.nblocks
        )
