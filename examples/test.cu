
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

__device__ double uniform_rand(curandState *state, double a,
                                double b) {
return a+(b-a)*curand_uniform_double(state);
}

} // extern C


extern "C" {
__global__ void draw_runtimes(double *tauR,
    curandState *state,
    const int N)
{ // for loop allows more particles than threads.
for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
    tid += blockDim.x * gridDim.x) {
    tauR[tid] = 1.0;
    }
}

// evolution of RTPs in 2D.
__global__ void
update(double *__restrict__ xold,       // old position in x
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
       double dt,   // time step
       int N
       )
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
x[tid] = xold[tid] + dt * 0 + dt * U0 * cos(thetaold[tid]);

y[tid] = yold[tid] + dt * 0 + dt * U0 * sin(thetaold[tid]);

// theta is only changing due to the vorticity of the flow at this stage!
theta[tid] = thetaold[tid] + dt * 0;
// need to update time since last tumble.
tau[tid] += dt;

// compute radial coordinate of the particle
double rp = sqrt(x[tid]*x[tid]+y[tid]*y[tid]);
if (rp > 1.0)
{
    x[tid] *= 1.0/rp;
    y[tid] *= 1.0/rp;
}
// set old to new.
xold[tid] = x[tid];
yold[tid] = y[tid];
thetaold[tid] = theta[tid];
} // end thread loop.
} // end bd kernel.
}
// initialize particle configuration
extern "C" {
__global__ void init_config(curandState *state,
                            double *x,
                            double *y,
                            double *theta,
                            const int N) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    x[tid] = 0;
    y[tid] = 0;
    theta[tid] = 0 + 6.283185307179586*curand_uniform_double(&state[tid]);
  }
} // end function
} // extern C
    