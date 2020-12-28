from ald.abp.abpkernels import AbstractABPKernel
from jinja2 import Template
from ald.core.config import AbstractConfig
import pycuda.gpuarray as gpuarray
import numpy as np
from ald.core.particle import ABP
from ald.core.boundary import Box
import h5py


class RheologyParams:
    """Compute simulation parameters from dimensionless quantities."""

    def __init__(self, gamma=1, Pes=1, Pe=1, alpha=1, dtscale=1e-4, tfscale=2000):
        self.gamma = gamma
        self.Pes = Pes
        self.Pe = Pe
        self.DT = 1.0
        # alpha = a/b
        self.alpha = alpha
        self.dtscale = dtscale
        self.tfscale = tfscale
        if alpha > 1:
            self.b = 1.0
            self.a = self.alpha * self.b
        else:
            self.a = 1.0
            self.b = self.a / self.alpha
        self.Rc = self.a + self.b
        # probe speed
        self.U1 = self.Pe * self.DT / self.Rc
        self.U2 = self.Pes * self.DT / self.Rc
        self.delta = self.Rc / self.gamma
        self.tauR = self.delta ** 2 / self.DT
        self.ell = self.U2 * self.tauR
        self.DR = 1 / self.tauR
        self.set_timescales()
        self.set_boxsize()

    def set_timescales(self):
        ts = []
        lmin = min(self.a, self.b)
        lmax = max(self.a, self.b)
        #
        Umin = min(self.U1, self.U2)
        Umax = max(self.U1, self.U2)
        # small advection time
        if np.abs(Umax) > 1e-8:
            ts.append(lmin / Umax)
        # large advection time
        if np.abs(Umin) > 1e-8:
            ts.append(lmax / Umin)
        # diffusive time
        ts.append(lmin ** 2 / self.DT)
        ts.append(lmax ** 2 / self.DT)
        # reorientation time
        # only need this for active particles
        if self.U2 != 0.0:
            ts.append(self.tauR)
        # determine min and max time scales.
        self.minscale = np.min(ts)
        self.maxscale = np.max(ts)
        # set time steps
        self.dt = self.dtscale * self.minscale
        self.tf = self.tfscale * self.maxscale
        self.Nt = int(self.tf / self.dt)
        # recompute tf
        self.tf = self.Nt * self.dt
        return None

    def is_wakecutoff(self):
        """Determine whether to use wake cutoff BC."""
        if self.Pe >= self.Pes * 10 and self.Pe >= 10:
            # if probe speed is much larger than ABP
            return True
        else:
            return False

    def set_boxsize(self):
        # time scales required to determine box size
        # time required for ABP to diffuse back
        ts = []
        ts.append((2 * self.a + 2 * self.b) ** 2 / self.DT)
        # time for ABP to swim back
        # only needed when U2 is not zero
        if self.U2 != 0:
            ts.append((2 * self.a + 2 * self.b) / self.U2)
        # length scale containers
        xs = []
        ys = []
        # box size should be larger than the distance
        # the probe travels to avoid going into the wake of the periodic images.
        tmax = np.max(ts)
        xs.append(20 * self.Rc)
        ys.append(20 * self.Rc)
        # delta
        delta = np.sqrt(self.DT * self.tauR)
        xs.append(30 * delta)
        ys.append(30 * delta)
        # run length scale
        ell_scale = 20
        xs.append(ell_scale * self.ell)
        ys.append(ell_scale * self.ell)
        self.W = np.max(ys)
        self.L = np.max(xs)
        # wake cutoff
        self.px = 2 * self.Rc
        self.py = self.W / 2.0
        # not cuting off wake
        if not self.is_wakecutoff():
            # determine the wake length and add it to lengths
            Lwake = 10 * self.U1 * tmax
            xs.append(Lwake)
            self.L = np.max(xs)
            # probe location
            self.px = self.L / 2

    @classmethod
    def from_passive(cls, Pe=1, alpha=1, dtscale=1e-4, tfscale=2000):
        gamma = 1.0
        Pes = 0.0
        return cls(
            gamma=gamma, Pes=Pes, Pe=Pe, alpha=alpha, dtscale=dtscale, tfscale=tfscale
        )


class RheologyConfig(AbstractConfig):
    def __init__(self, params, N=204800):
        if not isinstance(params, RheologyParams):
            raise TypeError("invalid type: {}".format(type(params)))
        self.params = params
        particle = ABP(
            U0=self.params.U2, tauR=self.params.tauR, DT=self.params.DT, a=self.params.a
        )
        domain = Box(left=0, right=self.params.L, bottom=0, top=self.params.W)
        super().__init__(particle, domain, N, self.params.dt, self.params.Nt)
        # need to add additional arrays to save collision
        self.dx = gpuarray.GPUArray(N, dtype=np.float64)
        self.dy = gpuarray.GPUArray(N, dtype=np.float64)

    def save2h5(self, file):
        """Save particle, domain simulation attributes to hdf5"""
        with h5py.File(file, "r+") as f:
            # write attributes only, not data arrays.
            for attr, value in vars(self.params).items():
                if np.isscalar(value) and np.isreal(value):
                    f.attrs[attr] = value
            # save additional parameter.s
            f.attrs["N"] = self.N


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

y[tid] += dt * U0 * sin(theta[tid])+ {{yB}}*curand_normal_double(&state[tid]);

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
}// end extern C
"""
)

periodic = Template(
    """
// Periodic boundary condition
while (x[tid] < 0) {
    x[tid] += {{L}};
}
while (x[tid] > {{L}}) {
    x[tid] -= {{L}};
}
// periodic in y
while (y[tid] < 0) {
    y[tid] += {{W}};
}
while (y[tid] > {{W}}) {
    y[tid] -= {{W}};
}
"""
)
wake_cutoff = Template(
    """
// particles leaving from the left/right is removed and added back in as a new
// particle with bulk properties.
if (x[tid] < 0 || x[tid] > {{L}}) {
    // enters from the right
    x[tid] = {{L}};
    // random y,z position
    y[tid] = curand_uniform_double(&state[tid]) * {{W}};
    // random orientation
    theta[tid] = curand_uniform_double(&state[tid]) * 2 * PI;
}
// periodic in y
if (y[tid] < 0) {
    y[tid] += {{W}};
}
if (y[tid] > {{W}}) {
    y[tid] -= {{W}};
}
"""
)


class RheologyKernel(AbstractABPKernel):
    """Cuda kernel for fixed velocity microrheology in 2D."""

    def __init__(self, params):
        if not isinstance(params, RheologyParams):
            raise TypeError("invalid type: {}".format(type(params)))
        if params.is_wakecutoff():
            self.wakecutoff = True
        else:
            self.wakecutoff = False
        # no additional args
        arg_list = ""
        super().__init__(arg_list)
        # need to modify the kernel_template to use the custom one
        # instead of the simple ABP one
        self.kernel_tempalte = rheology_kernel_template

    def _periodic_bc(self, cfg, flow):
        code = periodic.render(L=cfg.params.L, W=cfg.params.W)
        return code

    def _wake_cutoff_bc(self, cfg, flow):
        code = wake_cutoff.render(L=cfg.params.L, W=cfg.params.W)
        return code

    def generate_cuda_code(self, cfg, flow):
        if self.wakecutoff:
            code = self._wake_cutoff_bc(cfg, flow)
        else:
            code = self._periodic_bc(cfg, flow)
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
            np.float64(cfg.params.U2),
            np.float64(cfg.params.U1),
            np.float64(cfg.params.a),
            np.float64(cfg.params.b),
            np.float64(cfg.params.px),
            np.float64(cfg.params.py),
            np.float64(cfg.params.dt),
            np.int32(cfg.N),
            block=(threads, 1, 1),
            grid=(blocks, 1),
        )
        return None
