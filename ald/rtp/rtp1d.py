# A 1D model with no-flux walls at ends.
# in this case, the orientation becomes +-1
from jinja2 import Template


template1DRTP = Template(
    """
extern "C" {
// evolution of RTPs in 1D.
__global__ void
bd_rtp(double *__restrict__ x,          // position x
       double *__restrict__ direction,      // +1 or -1
       curandState *__restrict__ state, // RNG state
       double *__restrict__ tauR, // reorientation time for each active particle
       double *__restrict__ tau,  // time since last tumble
       double U0,                 // swim speed
       double L,    // simulation box length in x
       double dt,   // time step
       int N,       // total number of active particles.
       )

{
    // for loop allows more particles than threads.
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    // need to tumble
    if (tau[tid] >= tauR[tid]) {
      // tumbles between +1 and -1.
      direction[tid] *= -1;
      // reset time since last tumble to zero.
      tau[tid] = 0.0;
      // after tumbling, need to draw a new tumbling time.
      //{{runtimefun}}(&state[tid], tauavg, alpha);
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


rtp_kernel1d_rendered = template1DRTP.render(
    runtime="pareto_runtime(&state[tid], 1, 1.2)"
)
