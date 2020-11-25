from os.path import isfile
from typing import ValuesView
from numpy.lib.arraysetops import isin
import pycuda.autoinit
import pycuda.curandom
import pycuda.compiler as compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from jinja2 import Template
import os
import h5py


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

// initialize particle configuration
__global__ void init_config(curandState *state, double *x, double *y,
                            double *theta, double L, double H, const int N) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    x[tid] = 0.0;
    y[tid] = (curand_uniform_double(&state[tid]) - 0.5) * H;
    theta[tid] = curand_uniform_double(&state[tid]) * 2 * PI;
  }
} // end function

} // extern C
"""


template = Template(
    """
extern "C" {
// evolution of RTPs in 2D.
__global__ void
bd_rtp(double *__restrict__ xold,       // old position in x
       double *__restrict__ yold,       // old position in y
       double *__restrict__ thetaold,   // old orientation angle
       double *__restrict__ x,          // position x
       double *__restrict__ y,          // position y
       double *__restrict__ theta,      // orientation angle
       int *__restrict__ passx,         // total boundary crossings in x
       int *__restrict__ passy,         // total boundary crossings in y
       curandState *__restrict__ state, // RNG state
       double *__restrict__ tauR, // reorientation time for each active particle
       double *__restrict__ tau,  // time since last reorientation.
       double U0,                 // ABP swim speed
       double uf,                 // flow speed centerline
       double L,    // simulation box length in x
       double H,    // channel width, also simulation box size in y
       double dt,   // time step
       int N,       // total number of active particles.
       double tauavg, // average runtime of the runtime distribution.
       double alpha) // exponent in Pareto distribution

{
    // for loop allows more particles than threads.
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    // need to tumble
    if (tau[tid] >= tauR[tid]) {
      // the orientation needs to change in a discrete fashion due to
      // tumbling. pick a new orientation uniformly between 0 and 2pi
      thetaold[tid] = curand_uniform_double(&state[tid]) * 2.0 * PI;
      // reset time since last tumble to zero.
      tau[tid] = 0.0;
      // after tumbling, need to draw a new tumbling time.
      // tauR[tid] = pareto_runtime(&state[tid], tauR, alpha);
      tauR[tid] = {{runtimefun}}(&state[tid], tauavg, alpha);
    }
    // next update the position and orientation
    x[tid] =
        xold[tid] + dt * uf * (1.0 - 4.0 * yold[tid] * yold[tid] / (H * H)) +
        dt * U0 * cos(thetaold[tid]);

    y[tid] = yold[tid] + dt * U0 * sin(thetaold[tid]);

    // theta is only changing due to the vorticity of the flow at this stage!
    theta[tid] = thetaold[tid] + dt * 4 * uf * yold[tid] / (H * H);
    // need to update time since last tumble.
    tau[tid] += dt;

    // check and record boundary crossing.
    // periodic in x
    // boundary crossings are accumulated.
    // x in [-L/2,L/2]
    if (x[tid] < -L / 2.0) {
      {{left_bc}}
      passx[tid] += -1;
    } else if (x[tid] > L / 2.0) {
      {{right_bc}}
      passx[tid] += 1;
    }
    // BC in y direction.
    if (y[tid] > H / 2.0) {
      {{ top_bc }}
      passy[tid] += 1;
    } else if (y[tid] < -H / 2.0) {
      {{ bottom_bc }}
      passy[tid] += -1;
    }
    // set old to new.
    xold[tid] = x[tid];
    yold[tid] = y[tid];
    thetaold[tid] = theta[tid];
  } // end thread loop.
} // end bd kernel.
}
"""
)


class AbstractRTP:
    def __init__(self, U0=1.0, tauR=1.0):
        # tauR is the mean runtime.
        self.tauR = tauR
        self.U0 = U0


class RTP(AbstractRTP):
    """Constant runtime RTP."""

    def __init__(self, U0=1.0, tauR=1.0):
        super().__init__(U0=U0, tauR=tauR)

    def __repr__(self):
        return "RTP(U0 = {:.3f}, tauR= {:.3f})".format(self.U0, self.tauR)


class Pareto(AbstractRTP):
    def __init__(self, U0=1.0, tauR=1.0, alpha=1.2):
        super().__init__(U0, tauR)
        self.alpha = alpha
        self.taum = (alpha - 1) * tauR / alpha

    def __repr__(self):
        return (
            "Pareto(U0 = {:.3f}, tauR= {:.3f}, alpha = {:.2f}, taum = {:.3f})".format(
                self.U0, self.tauR, self.alpha, self.taum
            )
        )


class BC:
    pass


class NoFlux(BC):
    def __repr__(self):
        return "NoFlux()"


class Periodic(BC):
    def __repr__(self):
        return "Periodic()"


class Box:
    """Rectangular simulation box"""

    def __init__(
        self,
        L=1.0,
        H=1.0,
        left_bc=Periodic(),
        right_bc=Periodic(),
        bottom_bc=NoFlux(),
        top_bc=NoFlux(),
    ):
        self.L = L
        self.H = H
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.bottom_bc = bottom_bc
        self.top_bc = top_bc

    def __repr__(self):
        return "Box(L={:.3f}, H={:.3f}, left_bc={}, right_bc={}, bottom_bc={},top_bc={})".format(
            self.L, self.H, self.left_bc, self.right_bc, self.bottom_bc, self.top_bc
        )


class Configuration:
    def __init__(
        self,
        rtp=RTP(),
        L=1.0,
        H=1.0,
        left_bc=Periodic(),
        right_bc=Periodic(),
        bottom_bc=NoFlux(),
        top_bc=NoFlux(),
        uf=1.0,
        N=1000000,
    ):
        self.rtp = rtp
        self.box = Box(
            L=L,
            H=H,
            left_bc=left_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            top_bc=top_bc,
        )
        self.uf = uf
        self.N = int(N)
        # current configuration
        self.x = gpuarray.GPUArray(N, dtype=np.float64)
        self.y = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta = gpuarray.GPUArray(N, dtype=np.float64)
        # configuration at previous time step
        self.x_old = gpuarray.GPUArray(N, dtype=np.float64)
        self.y_old = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta_old = gpuarray.GPUArray(N, dtype=np.float64)
        # initial configuration,
        self.x0 = gpuarray.GPUArray(N, dtype=np.float64)
        self.y0 = gpuarray.GPUArray(N, dtype=np.float64)
        self.theta0 = gpuarray.GPUArray(N, dtype=np.float64)
        # curandstate array.
        sizeof_state = pycuda.characterize.sizeof(
            "curandStateXORWOW", "#include <curand_kernel.h>"
        )
        self.state = cuda.mem_alloc(N * sizeof_state)
        # !!!!!!!!!!!!np.int64 will cause problems!.
        self.passx = gpuarray.GPUArray(N, dtype=np.int32)
        self.passy = gpuarray.GPUArray(N, dtype=np.int32)
        self.tauR = gpuarray.GPUArray(N, dtype=np.float64)
        self.tau = gpuarray.GPUArray(N, dtype=np.float64)

    @classmethod
    def from_freespace(cls, rtp=RTP(), L=1.0, N=1000000):
        return cls(
            rtp=rtp,
            L=L,
            H=L,
            left_bc=Periodic(),
            right_bc=Periodic(),
            bottom_bc=Periodic(),
            top_bc=Periodic(),
            uf=0.0,
            N=N,
        )

    @classmethod
    def from_Poiseuille(cls, rtp=RTP(), H=1.0, uf=1.0, N=1000000):
        return cls(
            rtp=rtp,
            L=H,
            H=H,
            left_bc=Periodic(),
            right_bc=Periodic(),
            bottom_bc=NoFlux(),
            top_bc=NoFlux(),
            uf=uf,
            N=N,
        )


class Simulator:
    """Langevin simulation of RTPs in 2D (channel or freespace.)"""

    def __init__(self, config):
        if not isinstance(config, Configuration):
            raise ValueError("{} not an instance of Configuration.".format(config))
        self.rtp = config.rtp
        self.box = config.box
        self.cuda_code = cuda_code
        # do cuda compilation
        self.compile_cuda()
        # cuda launch parameters
        # cuda kernel launch parameter
        self.threadsPerBlock = 512
        self.nblocks = config.N // self.threadsPerBlock + 1
        if self.nblocks > 500:
            self.nblocks = 500

    def generate_bdrtp(self):
        if isinstance(self.rtp, RTP):
            runtimefun = "constant_runtime"
        elif isinstance(self.rtp, Pareto):
            runtimefun = "pareto_runtime"
        else:
            raise NotImplementedError()

        if isinstance(self.box.bottom_bc, NoFlux):
            bottom_bc = "y[tid] = -H / 2.0;"
        elif isinstance(self.box.bottom_bc, Periodic):
            bottom_bc = "y[tid] += H;"
        else:
            raise NotImplementedError()

        if isinstance(self.box.top_bc, NoFlux):
            top_bc = "y[tid] = H / 2.0;"
        elif isinstance(self.box.top_bc, Periodic):
            top_bc = "y[tid] -= H;"
        else:
            raise NotImplementedError()

        if isinstance(self.box.left_bc, NoFlux):
            left_bc = "x[tid] = -L / 2.0;"
        elif isinstance(self.box.left_bc, Periodic):
            left_bc = "x[tid] += L;"
        else:
            raise NotImplementedError()

        if isinstance(self.box.right_bc, NoFlux):
            right_bc = "x[tid] = L / 2.0;"
        elif isinstance(self.box.right_bc, Periodic):
            right_bc = "x[tid] -= L;"
        else:
            raise NotImplementedError()

        # render the bd_rtp template
        bd_rtp = template.render(
            runtimefun=runtimefun,
            top_bc=top_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
            right_bc=right_bc,
        )
        return bd_rtp

    def log2file(self, log):
        """Log cuda sourcecode to a cuda file."""
        if not isinstance(log, str):
            raise ValueError("invalid filename {}".format(log))
        # log to a file
        # if the filename already exist, append _new.cu
        if isfile(log):
            log += "new.cu"
        with open(log, "w") as f:
            f.write(self.cuda_code)
        print("cuda source code saved at {}".format(log))
        return None

    def compile_cuda(self, log=None):
        """Compile cuda source code"""
        # with open(self.src, "r") as file:
        #     code = file.read()
        # add bd_rtp templated code
        bd_rtp = self.generate_bdrtp()
        # combine cuda source codes
        # make sure this function can be run multiple times
        self.cuda_code = cuda_code
        # append bd_rtp.
        self.cuda_code += bd_rtp
        if log is not None:
            self.log2file(log)

        module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
        # get functions from cuda module
        self.bdrtp = module.get_function("bd_rtp")
        self.initrand = module.get_function("initrand")
        self.init_config = module.get_function("init_config")
        self.draw_pareto_runtimes = module.get_function("draw_pareto_runtimes")

        return None

    def launch_kernel(self, func, *args):
        """Launch cuda kernel func"""
        func(
            *args,
            block=(self.threadsPerBlock, 1, 1),
            grid=(self.nblocks, 1),
        )
        return None

    def initialize(self, cfg):
        """Initialize the particle and simulation configurations."""
        # initialize RNG
        self.launch_kernel(self.initrand, cfg.state, np.int32(cfg.N))
        # initialize particle configuration
        self.launch_kernel(
            self.init_config,
            cfg.state,
            cfg.x0,
            cfg.y0,
            cfg.theta0,
            np.float64(self.box.L),
            np.float64(self.box.H),
            np.int32(cfg.N),
        )

        # need to fill x,y,theta old too
        cfg.x_old = cfg.x0.copy()
        cfg.y_old = cfg.y0.copy()
        cfg.theta_old = cfg.theta0.copy()
        # need to initialize passx and passy to 0
        cfg.passx.fill(0)
        cfg.passy.fill(0)
        # similarly, time since last reorientation is zero,.
        cfg.tau.fill(0.0)
        # need to initialize the runtime for each particle.
        if isinstance(self.rtp, RTP):
            # constant runtime.
            cfg.tauR.fill(self.rtp.tauR)
        elif isinstance(self.rtp, Pareto):
            # Pareto distributed runtimes.
            self.launch_kernel(
                self.draw_pareto_runtimes,
                cfg.tauR,
                cfg.state,
                np.int32(cfg.N),
                np.float64(self.rtp.tauR),
                np.float64(self.rtp.alpha),
            )
        else:
            raise NotImplementedError()

        return None

    def simulate(self, cfg):
        """Run Langevin simulation"""
        Nt = 1000000
        t = 0.0
        dt = 1e-4
        if isinstance(cfg.rtp, Pareto):
            alpha = cfg.rtp.alpha
        else:
            # in this case, its a dummy variable.
            alpha = 0.0

        for i in range(Nt):
            self.launch_kernel(
                self.bdrtp,
                cfg.x_old,
                cfg.y_old,
                cfg.theta_old,
                cfg.x,
                cfg.y,
                cfg.theta,
                cfg.passx,
                cfg.passy,
                cfg.state,
                cfg.tauR,
                cfg.tau,
                np.float64(cfg.rtp.U0),
                np.float64(cfg.uf),
                np.float64(cfg.box.L),
                np.float64(cfg.box.H),
                np.float64(dt),
                np.int32(cfg.N),
                np.float64(cfg.rtp.tauR),
                np.float64(alpha),
            )
            if i % 50000 == 0:
                print(gpuarray.sum((cfg.x + cfg.passx * cfg.box.L) / cfg.N))
            t += dt


# rtp = RTP()
# cfg = Configuration.from_Poiseuille(rtp=rtp, uf=0.1, N=204800)
# # cfg = Configuration.from_freespace(rtp=rtp, N=204800)
# simulator = Simulator(cfg)
# simulator.initialize(cfg)
# simulator.simulate(cfg)
