from ald.core.external_velocity import ZeroVelocity
from jinja2 import Template
from abc import ABC, abstractmethod
import pycuda.gpuarray as gpuarray
import numpy as np
from ald.core.kernel import AbstractKernel
from ald.core.bc import (
    LeftNoFlux,
    RightNoFlux,
    BottomNoFlux,
    TopNoFlux,
    LeftPBC,
    RightPBC,
    BottomPBC,
    TopPBC,
)


abp_kernel_template = Template(
    """
extern "C" {
// evolution of ABPs in 2D
__global__ void
update(double *__restrict__ xold,       // old position in x
       double *__restrict__ yold,       // old position in y
       double *__restrict__ thetaold,   // old orientation angle
       double *__restrict__ x,          // position x
       double *__restrict__ y,          // position y
       double *__restrict__ theta,      // orientation angle
       int *__restrict__ passx,         // total boundary crossings in x
       int *__restrict__ passy,         // total boundary crossings in y
       curandStatePhilox4_32_10_t *__restrict__ state, // RNG state
       double U0,                 // ABP swim speed
       double dt,   // time step
       int N{{arg_list}}
       )

{
// for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
    tid += blockDim.x * gridDim.x) {
// can generate two random numbers at the same time
double2 rn2 = curand_normal2_double(&state[tid]);
// next update the position and orientation
x[tid] = xold[tid] + dt * {{ux}} + dt * U0 * cos(thetaold[tid]) + {{xB}}*rn2.x;

y[tid] = yold[tid] + dt * {{uy}} + dt * U0 * sin(thetaold[tid])+ {{yB}}*rn2.y;

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


class AbstractABPKernel(AbstractKernel):
    def __init__(self, arg_list):
        # abp kernel template
        self.kernel_tempalte = abp_kernel_template
        super().__init__(arg_list)
        # store domain and flow info

    def compute_diffusion_amplitue(self, cfg):
        """compute xB, yB and thetaB"""
        xB = np.sqrt(2 * cfg.dt * cfg.particle.DT)
        thetaB = np.sqrt(2 * cfg.dt * cfg.particle.DR)
        return xB, xB, thetaB


class ABPFreespaceKernel(AbstractABPKernel):
    """Doubly periodic BC."""

    def __init__(self):
        # no additional args
        arg_list = ""
        super().__init__(arg_list)

    def generate_cuda_code(self, cfg, flow):
        bcs = [LeftPBC(), RightPBC(), BottomPBC(), TopPBC()]
        code = ""
        for bc in bcs:
            code += bc.cuda_code(cfg.domain).__str__()
            # add a new line
            code += "\n"
        # now ready to render kernel template
        # compute Brownian displacement amplitudes
        xB, yB, thetaB = self.compute_diffusion_amplitue(cfg)
        kernel = self.kernel_tempalte.render(
            arg_list=self.arg_list,
            ux=flow.ux,
            uy=flow.uy,
            omega=flow.omega,
            xB=xB,
            yB=yB,
            thetaB=thetaB,
            code=code,
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
            np.int32(cfg.N),
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )
        return None


class ABPChannelKernel(AbstractABPKernel):
    """No flux boudnaries at top and bottom"""

    def __init__(self, displacement=False):
        # no additional args
        # whether to record collision displacements on the walls.
        self.displacement = displacement
        if self.displacement:
            arg_list = ",\ndouble *dy1,\n double *dy2"
        else:
            arg_list = ""
        super().__init__(arg_list)

    def generate_cuda_code(self, cfg, flow):
        bcs = [
            LeftPBC(),
            RightPBC(),
            BottomNoFlux(displacement=self.displacement, displacement_var="dy1"),
            TopNoFlux(displacement=self.displacement, displacement_var="dy2"),
        ]
        # need to add containers to cfg
        if self.displacement:
            setattr(cfg, "dy1", gpuarray.GPUArray(cfg.N, np.float64))
            setattr(cfg, "dy2", gpuarray.GPUArray(cfg.N, np.float64))
        code = ""
        for bc in bcs:
            code += bc.cuda_code(cfg.domain).__str__()
            # add a new line
            code += "\n"
        # now ready to render kernel template
        # compute Brownian displacement amplitudes
        xB, yB, thetaB = self.compute_diffusion_amplitue(cfg)
        kernel = self.kernel_tempalte.render(
            arg_list=self.arg_list,
            ux=flow.ux,
            uy=flow.uy,
            omega=flow.omega,
            xB=xB,
            yB=yB,
            thetaB=thetaB,
            code=code,
        )
        return kernel

    def update(self, func, cfg, threads, blocks):
        if self.displacement:
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
                np.int32(cfg.N),
                cfg.dy1,
                cfg.dy2,
                block=(threads, 1, 1),
                grid=(blocks, 1),
            )
        else:
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
                np.int32(cfg.N),
                block=(threads, 1, 1),
                grid=(blocks, 1),
            )
        return None
