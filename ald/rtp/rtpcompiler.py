from typing_extensions import runtime
from ald.core.compiler import AbstractCompiler
from ald.core.particle import AbstractRTP, RTP, Pareto
from ald.core.external_velocity import ExternalVelocity, ZeroVelocity, Poiseuille
from ald.core.ic import Point, Uniform, InitialConfig
from ald.core.boundary import AbstractDomain, Box
from jinja2 import Template
import os
import pycuda.compiler as compiler
from .rtpkernels import AbstractRTPKernel


class RTPCompiler(AbstractCompiler):
    """Cuda code compiler for RTPs."""

    def __init__(
        self,
        kernel,
        cfg,
        flow=ZeroVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(cfg.particle, AbstractRTP):
            raise TypeError()
        if not isinstance(kernel, AbstractRTPKernel):
            raise TypeError()

        super().__init__(kernel, cfg, flow=flow, ic=ic)

    def generate_cuda_code(self, cfg, flow):
        # combine cuda source codes
        # make sure this function can be run multiple times
        # do not touch self.cuda_code_base.
        # get the base code
        code = self.cuda_code_base
        # runtime generation kernel
        code += self.particle.runtime_device_code
        # append bd_code.
        code += self.kernel.generate_cuda_code(cfg, flow)
        # append initial condition kernel
        code += self.ic.cuda_code
        return code

    def compile(self, log=None):
        """Compile cuda source code"""
        module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
        # get functions from cuda module
        self.update = module.get_function("update")
        self.initrand = module.get_function("initrand")
        self.init_config = module.get_function("init_config")
        self.draw_runtimes = module.get_function("draw_runtimes")
