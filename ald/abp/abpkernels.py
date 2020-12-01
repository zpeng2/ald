from ald.core.external_velocity import ZeroVelocity
from jinja2 import Template
from abc import ABC, abstractmethod
import pycuda.gpuarray as gpuarray
import numpy as np


abp_kernel_template = Template(
    """
extern "C" {
// evolution of ABPs in 2D
__global__ void
update_abp(double *__restrict__ xold,       // old position in x
       double *__restrict__ yold,       // old position in y
       double *__restrict__ thetaold,   // old orientation angle
       double *__restrict__ x,          // position x
       double *__restrict__ y,          // position y
       double *__restrict__ theta,      // orientation angle
       int *__restrict__ passx,         // total boundary crossings in x
       int *__restrict__ passy,         // total boundary crossings in y
       curandState *__restrict__ state, // RNG state
       double U0,                 // ABP swim speed
       double dt,   // time step
       {{arg_list}}
       int N)

{
    // for loop allows more particles than threads.
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    // next update the position and orientation
    x[tid] = xold[tid] + dt * {{ux}} + dt * U0 * cos(thetaold[tid]) + {{xB}}*curand_normal_double(&state[tid]);

    y[tid] = yold[tid] + dt * {{uy}} + dt * U0 * sin(thetaold[tid])+ {{yB}}*curand_normal_double(&state[tid]);

    // update orientation
    theta[tid] = thetaold[tid] + dt*{{omega}} + {{thetaB}}*curand_normal_double(&state[tid]);

    {{code}}
    // set old to new.
    xold[tid] = x[tid];
    yold[tid] = y[tid];
    thetaold[tid] = theta[tid];
  } // end thread loop.
} // end bd kernel.
}
"""
)


class AbstractABPKernel(ABC):
    """SUbclass should properly define the kernel and provide interface for calling the kernel in python"""

    def __init__(self, kernel_code):
        self.kernel_code = kernel_code

    @abstractmethod
    def render_cuda_kernel(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


planar_wall_code = Template(
    """
if (x[tid] < {{left}})
{
    // displacement of the ABP. (ABP has no size.)
    dx[tid] = {{left}} - x[tid];
    x[tid] = {{left}};
}
if (x[tid] > {{right}})
{
    // randomizing condition.
    y[tid] = uniform_rand(&state[tid], {{bottom}}, {{top}});
    theta[tid] = uniform_rand(&state[tid], 0, 2*PI);
    x[tid] = {{right}};
}
if (y[tid] < {{bottom}})
{
    y[tid] += {{Ly}};
    passy[tid] += -1;
}
if (y[tid] > {{top}})
{
    y[tid] -= {{Ly}};
    passy[tid] +=1;
}
"""
)


class PlanarWallKernel(AbstractABPKernel):
    def __init__(self, cfg, flow=ZeroVelocity()):
        self.kernel_template = abp_kernel_template
        self.code_template = planar_wall_code
        # generate kernel
        kernel = self.render_cuda_kernel(cfg, flow)
        super().__init__(kernel)

    def render_cuda_kernel(self, cfg, flow):
        """render kernel"""
        # need to add a new array
        setattr(cfg, "dx", gpuarray.GPUArray(cfg.N, np.float64))
        arg_list = "double *dx,"
        # render bc
        bc = self.code_template.render(
            left=cfg.domain.left,
            right=cfg.domain.right,
            top=cfg.domain.top,
            bottom=cfg.domain.bottom,
            Ly=cfg.domain.Ly,
        )
        # render kernel
        # compute brownian amplitude
        xB = np.sqrt(2 * cfg.dt * cfg.particle.DT)
        thetaB = np.sqrt(2 * cfg.dt * cfg.particle.DR)
        kernel = self.kernel_template.render(
            arg_list=arg_list,
            ux=flow.ux,
            uy=flow.uy,
            omega=flow.omega,
            xB=xB,
            yB=xB,
            thetaB=thetaB,
            code=bc,
        )
        return kernel

    def update(self, func, cfg, threads, blocks):
        func(
            cfg.x_old,
            cfg.y_old,
            cfg.theta_old,
            cfg.x,
            cfg.y,
            cfg.theta,
            cfg.passx,
            cfg.passy,
            cfg.state,
            np.float64(cfg.particle.U0),
            np.float64(cfg.dt),
            cfg.dx,
            np.int32(cfg.N),
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )
        return None
