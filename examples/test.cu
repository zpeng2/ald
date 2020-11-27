
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
                            double *theta, double Lx, double Ly, const int N) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    x[tid] = 0.0;
    y[tid] = (curand_uniform_double(&state[tid]) - 0.5) * Ly;
    theta[tid] = curand_uniform_double(&state[tid]) * 2 * PI;
  }
} // end function

} // extern C

extern "C" {
// evolution of RTPs in 2D.
__global__ void
update_rtp(double *__restrict__ xold,       // old position in x
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
       double Lx,    // simulation box length in x
       double Ly,    // channel width, also simulation box size in y
       double dt,   // time step
       int N)

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
      tauR[tid] = 1.0;
    }
    // next update the position and orientation
    x[tid] = xold[tid] + dt * 1.0*(1.0-4.0*yold[tid]*yold[tid]/(1.0*1.0)) + dt * U0 * cos(thetaold[tid]);

    y[tid] = yold[tid] + dt * 0 + dt * U0 * sin(thetaold[tid]);

    // theta is only changing due to the vorticity of the flow at this stage!
    theta[tid] = thetaold[tid] + dt * 4*1.0*yold[tid]/(1.0*1.0);
    // need to update time since last tumble.
    tau[tid] += dt;

    // check and record boundary crossing.
    // periodic in x
    // boundary crossings are accumulated.
    // x in [-Lx/2,Lx/2]
    if (x[tid] < -Lx / 2.0) {
      x[tid] += 1.0;
      passx[tid] += -1;
    } else if (x[tid] > Lx / 2.0) {
      x[tid] -= 1.0;
      passx[tid] += 1;
    }
    // BC in y direction.
    if (y[tid] > Ly / 2.0) {
      y[tid] -= 1.0;
      passy[tid] += 1;
    } else if (y[tid] < -Ly / 2.0) {
      y[tid] += 1.0;
      passy[tid] += -1;
    }
    // set old to new.
    xold[tid] = x[tid];
    yold[tid] = y[tid];
    thetaold[tid] = theta[tid];
  } // end thread loop.
} // end bd kernel.
}