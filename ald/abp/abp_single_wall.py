from .abpkernels import AbstractABPKernel
from jinja2 import Template
import numpy as np
import pycuda.gpuarray as gpuarray


# A wall on the left of the simulation domain.
left_wall_bc = Template(
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


class ABPSingleWallKernel(AbstractABPKernel):
    def __init__(self):
        # additional arg to record the force on the wall.
        arg_list = ",\ndouble *dx"
        super().__init__(arg_list)
    def generate_cuda_code(self, cfg, flow):
        # need to add the new argument to cfg.
        setattr(cfg, "dx", gpuarray.GPUArray(cfg.N, np.float64))
        bc = left_wall_bc.render(left = cfg.domain.left, right = cfg.domain.right, bottom = cfg.domain.bottom, top = cfg.domain.top, Ly = cfg.domain.Ly)
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
            np.int32(cfg.N),
            cfg.dx,
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )
        return None






