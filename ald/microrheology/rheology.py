from ald.abp.abpkernels import AbstractABPKernel
from jinja2 import Template
from ald.core.bc import (
    LeftPBC,
    RightPBC,
    BottomPBC,
    TopPBC,
)
from ald.core.config import AbstractConfig
import pycuda.gpuarray as gpuarray
import numpy as np


class RheologyConfiger:
    """Compute simulation parameters from dimensionless quantities."""

    def __init__(self, gamma, Pes, Pe):
        self.gamma = gamma
        self.Pes = Pes
        self.Pe = Pe
        self.DT = 1.0
        self.a = 1.0
        self.b = 1.0
        self.Rc = self.a + self.b
        # probe speed
        self.U1 = self.Pe * self.DT / self.Rc
        self.U2 = self.Pes * self.DT / self.Rc
        self.delta = self.Rc / self.gamma
        self.tauR = self.delta ** 2 / self.DT
        self.ell = self.U2 * self.tauR
        self.DR = 1 / self.tauR

    def set_timescales(self):
        ts = []
        # small advection time
        if not (self.U1 == 0.0 and self.U2 == 0.0):
            ts.append(self.a / max(self.U1, self.U2))
        if self.U1 != 0.0 and self.U2 != 0.0:
            # large advection time
            ts.append(self.a / min(self.U1, self.U2))
        # diffusive time
        ts.append(self.a ** 2 / self.DT)
        # reorientation time
        # only need this for active particles
        if self.U2 != 0.0:
            ts.append(self.tauR)
        # determine min and max time scales.
        self.minscale = np.min(ts)
        self.maxscale = np.max(ts)

    def determine_wakecutoff(self):
        """Determine whether to use wake cutoff BC."""

    def configure(self):
        pass

    @classmethod
    def from_passive(cls, Pe):
        pass


class RheologyConfig(AbstractConfig):
    def __init__(self, particle, domain, a, b, px, py, Up, N, dt, Nt):
        super().__init__(particle, domain, N, dt, Nt)
        self.a = a
        self.b = b
        self.Up = Up
        self.px = px
        self.py = py
        # need to add additional arrays to save collision
        self.dx = gpuarray.GPUArray(N, dtype=np.float64)
        self.dy = gpuarray.GPUArray(N, dtype=np.float64)

    @classmethod
    def from_configer(cls, configer):
        """Construct from RheologyConfiger"""
        return cls(
            configer.particle,
            configer.domain,
            configer.a,
            configer.b,
            configer.px,
            configer.py,
            configer.Up,
            configer.N,
            configer.dt,
            configer.Nt,
        )


rheology_kernel_template = Template(
    """
extern "C" {
// Microrheology of ABPs in 2D. periodic BCs.
__global__ void
update(double *__restrict__ x,          // position x
       double *__restrict__ y,          // position y
       double *__restrict__ theta,      // orientation angle
       curandState *__restrict__ state, // RNG state
       double *__restrict__ dx,
       double *__restrict__ dy,
       double U0,                 // ABP swim speed
       double Up,   // speed of probe
       double a,    // probe radius
       double b,    // ABP radius
       double px,   // probe center location in x
       double py,   // probe location in y
       double dt,   // time step
       int N
       )

{
// for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
    tid += blockDim.x * gridDim.x) {
// next update the position and orientation
x[tid] += dt * -Up + dt * U0 * cos(theta[tid]) + {{xB}}*curand_normal_double(&state[tid]);

y[tid] += dt * U0 * sin(thetaold[tid])+ {{yB}}*curand_normal_double(&state[tid]);

// update orientation
theta[tid] += {{thetaB}}*curand_normal_double(&state[tid]);
// apply BC on the simulation box
{{code}}

// calculate HS displacements and move ABPs.
// In this method, the probe is fixed.
// make sure its far enough from the boundary.
// so that no need to check periodic images for overlapping.
// also that when ABPs are moved out, they are not outside the boundary.
// hard sphere algorithm
// calculate the cartesian distance between an ABP and the probe
// dx, dy are used multiple times to save memory allocation.
dx[tid] = x[tid] - px;
dy[tid] = y[tid] - py;
// need the actually distance for hard sphere move
double ds = sqrt(dx[tid] * dx[tid] + dy[tid] * dy[tid]);
// hard sphere move for ABPs
if (ds < a + b) {
    // compute ABP displacements
    dx[tid] *= ((a + b) / ds - 1);
    dy[tid] *= ((a + b) / ds - 1);
    // move ABPs.
    x[tid] += dx[tid];
    y[tid] += dy[tid];
} else {
    // no overlap.
    // important! need to reset to zero.
    dx[tid] = 0;
    dy[tid] = 0;
}
// here the radius ratio is needed.
// since the hard sphere movement is
// based on equal force
// 6 pi eta a dx1/dt +6 pi eta b dx2/dt=0
// a big probe moves less compared with
// a small bath particle.
} // end thread loop.
} // end bd kernel.
"""
)


class RheologyPeriodicKernel(AbstractABPKernel):
    """Doubly periodic BC."""

    def __init__(self):
        # no additional args
        arg_list = ""
        super().__init__(arg_list)
        # need to modify the kernel_template to use the custom one
        # instead of the simple ABP one
        self.kernel_tempalte = rheology_kernel_template

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
            ux=flow.ux,
            xB=xB,
            yB=yB,
            thetaB=thetaB,
            code=code,
        )
        return kernel

    def update(self, func, cfg, threads, blocks):
        func(
            cfg.x,
            cfg.y,
            cfg.theta,
            cfg.state,
            cfg.dx,
            cfg.dy,
            np.float64(cfg.particle.U0),
            np.float64(cfg.Up),
            np.float64(cfg.a),
            np.float64(cfg.b),
            np.float64(cfg.px),
            np.float64(cfg.py),
            np.float64(cfg.dt),
            np.int32(cfg.N),
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )
        return None
