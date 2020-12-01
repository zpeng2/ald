from ald.core.compiler import AbstractCompiler
from ald.core.particle import ABP
from ald.core.external_velocity import ExternalVelocity, ZeroVelocity, Poiseuille
from ald.core.ic import Point, Uniform, InitialConfig
from ald.core.boundary import AbstractDomain, Box
from jinja2 import Template
import os
import pycuda.compiler as compiler


class ABPCompiler(AbstractCompiler):
    """Cuda code compiler for RTPs."""

    def __init__(
        self,
        particle=ABP(),
        domain=Box.from_freespace(),
        flow=ZeroVelocity(),
        ic=InitialConfig(),
    ):
        if not isinstance(particle, ABP):
            raise TypeError("invalid particle type:{}".format(type(particle)))

        super().__init__(particle=particle, domain=domain, flow=flow, ic=ic)
        # render rtp kernel template.
        self._render_abp_kernel()
        # compile
        self.compile()

    def _render_rtp_kernel(self):
        """Run template renderer"""
        self.rtp_kernel_code = abp_kernel_template.render(
            runtime=self.particle.runtime_code,
            ux=self.flow.ux,
            uy=self.flow.uy,
            omega=self.flow.omega,
            bc=self.domain.bc,
        )

    @property
    def cuda_code(self):
        # combine cuda source codes
        # make sure this function can be run multiple times
        # do not touch self.cuda_code_base.
        # get the base code
        cuda_code = self.cuda_code_base
        # append bd_rtp.
        cuda_code += self.rtp_kernel_code
        # append initial condition kernel
        cuda_code += self.ic.cuda_code
        return cuda_code

    def compile(self, log=None):
        """Compile cuda source code"""
        module = compiler.SourceModule(self.cuda_code, no_extern_c=True, keep=False)
        # get functions from cuda module
        self.update_rtp = module.get_function("update_rtp")
        self.initrand = module.get_function("initrand")
        self.init_config = module.get_function("init_config")
        self.draw_pareto_runtimes = module.get_function("draw_pareto_runtimes")
