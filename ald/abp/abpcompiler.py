from ald.core.config import AbstractConfig
from typing_extensions import runtime
from ald.core.compiler import AbstractCompiler
from ald.core.particle import ABP
from ald.core.external_velocity import ExternalVelocity, ZeroVelocity, Poiseuille
from ald.core.ic import Point, Uniform, InitialConfig
from ald.core.boundary import AbstractDomain, Box
from jinja2 import Template
import os
import pycuda.compiler as compiler
from ald.abp.abpkernels import AbstractABPKernel, PlanarWallKernel


class ABPCompiler(AbstractCompiler):
    """Cuda code compiler for ABPs."""

    def __init__(
        self,
        kernel,
        cfg,
        flow=ZeroVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(kernel, AbstractABPKernel):
            raise TypeError()
        if not isinstance(cfg, AbstractConfig):
            raise TypeError()
        super().__init__(cfg, flow=flow, ic=ic)
        # keep a kernel
        self.kernel = kernel
        # compile
        self.compile()

    @property
    def cuda_code(self):
        # combine cuda source codes
        # make sure this function can be run multiple times
        # do not touch self.cuda_code_base.
        # get the base code
        cuda_code = self.cuda_code_base
        # append bd_code.
        cuda_code += self.kernel.kernel_code
        # append initial condition kernel
        cuda_code += self.ic.cuda_code
        return cuda_code

    def compile(self, log=None):
        """Compile cuda source code"""
        module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
        # get functions from cuda module
        self.update_abp = module.get_function("update_abp")
        self.initrand = module.get_function("initrand")
        self.init_config = module.get_function("init_config")
