from jinja2 import Template
import numpy as np


class AbstractIC:
    def __init__(self, cuda_code):
        if not isinstance(cuda_code, str):
            raise TypeError()
        self.cuda_code = cuda_code


class Point(AbstractIC):
    def __init__(self, loc):
        self.loc = loc
        cuda_code = "{}".format(self.loc)
        super().__init__(cuda_code)


class Uniform(AbstractIC):
    def __init__(self, a, b):
        # Describes uniform distribution , U[a,b]
        self.a = a
        self.b = b
        # map [0,1] to [a,b]
        delta = b - a
        cuda_code = "{0} + {1}*curand_uniform_double(&state[tid])".format(self.a, delta)
        super().__init__(cuda_code)


# initial condition cuda kernel.
ic_kernel = Template(
    """
// initialize particle configuration
extern "C" {
__global__ void init_config(curandState *state,
                            double *x,
                            double *y,
                            double *theta,
                            const int N) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
       tid += blockDim.x * gridDim.x) {
    x[tid] = {{x_ic}};
    y[tid] = {{y_ic}};
    theta[tid] = {{theta_ic}};
  }
} // end function
} // extern C
    """
)


class InitialConfig:
    def __init__(self, x=Point(0), y=Point(0), theta=Uniform(0, 2 * np.pi)):
        self._validate(x)
        self._validate(y)
        self._validate(theta)
        self.x = x
        self.y = y
        self.theta = theta
        self.cuda_code = ic_kernel.render(
            x_ic=self.x.cuda_code, y_ic=self.y.cuda_code, theta_ic=self.theta.cuda_code
        )

    def _validate(self, arg):
        """Validate argument to be type AbstractIC"""
        if not isinstance(arg, AbstractIC):
            raise TypeError("invalid argument type: {}".format(type(arg)))
