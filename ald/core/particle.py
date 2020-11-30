from abc import abstractmethod, ABC
from jinja2 import Template


class AbstractParticle:
    pass


class ABP(AbstractParticle):
    def __init__(self, U0=1.0, tauR=1.0, DT=1.0):
        self.U0 = U0
        self.tauR = tauR
        self.DR = 1 / tauR
        self.DT = DT

    def __repr__(self):
        return "ABP(U0 = {:.3f}, DR= {:.3f}, DT={:.3f})".format(
            self.U0,
            self.DR,
            self.DT,
        )


class AbstractRTP(AbstractParticle):
    def __init__(self, U0=1.0, tauR=1.0, runtime_code=None):
        # tauR is the mean runtime.
        self.tauR = tauR
        self.U0 = U0
        self.runtime_code = runtime_code


class RTP(AbstractRTP):
    """Constant runtime RTP."""

    def __init__(self, U0=1.0, tauR=1.0):
        runtime_code = "{}".format(tauR)
        super().__init__(U0=U0, tauR=tauR, runtime_code=runtime_code)

    def __repr__(self):
        return "RTP(U0 = {:.3f}, tauR= {:.3f})".format(self.U0, self.tauR)


exponential_runtime_device = """
// tauR is the mean runtime, inverse of lambda.
// https://en.wikipedia.org/wiki/Exponential_distribution#Generating_exponential_variates
extern "C" {
__device__ double exponential_runtime(curandState *state, double tauR) {
  double U = curand_uniform_double(state);
  return -log(U)*tauR;
}
} //extern C
"""


class ExponentialRTP(AbstractRTP):
    def __init__(self, U0=1.0, tauR=1.0):
        """tauR is the mean run time. PDF is lambda *exp(-lambda*x)
        The mean is 1/lambda = tauR.
        """
        runtime_code = "exponential_runtime(&state[tid], {})".format(tauR)
        super().__init__(U0=U0, tauR=tauR, runtime_code=runtime_code)
        self.lam = 1 / tauR
        # runtime device code that need to be added to the compiler.
        self.runtime_device_code = exponential_runtime_device

    def __repr__(self):
        return "RTP(U0 = {:.3f}, tauR= {:.3f}, lambda={:.3f})".format(
            self.U0, self.tauR, self.lam
        )


pareto_runtime_device = """
// generate Pareto run times.
// https://en.wikipedia.org/wiki/Pareto_distribution#Random_sample_generation
// taum is the minimum run time, alpha is the exponent
extern "C" {
__device__ double pareto_runtime(curandState *state, double tauR,
                                double alpha) {
// !! tauR is the mean runtime, which is taum*alpha/(alpha-1)
double taum = (alpha - 1.0) * tauR / alpha;
// curand_uniform_double is uniformly distributed in (0,1)
double U = curand_uniform_double(state);
return taum / pow(U, 1.0 / alpha);
}
}
"""


class Pareto(AbstractRTP):
    def __init__(self, U0=1.0, tauR=1.0, alpha=1.2):
        self.alpha = alpha
        self.taum = (alpha - 1) * tauR / alpha
        # see compiler for details.
        runtime_code = "pareto_runtime(&state[tid], {}, {})".format(tauR, alpha)
        super().__init__(U0=U0, tauR=tauR, runtime_code=runtime_code)

        # runtime device code that need to be added to the compiler.
        self.runtime_device_code = pareto_runtime_device

    def __repr__(self):
        return (
            "Pareto(U0 = {:.3f}, tauR= {:.3f}, alpha = {:.2f}, taum = {:.3f})".format(
                self.U0, self.tauR, self.alpha, self.taum
            )
        )
