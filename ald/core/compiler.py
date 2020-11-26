from abc import abstractmethod, ABC
import pycuda.gpuarray as gpuarray
import pycuda.curandom
import pycuda.compiler as compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np


class AbstractCompiler(ABC):
    @abstractmethod
    def compile(self):
        pass



class RTPCompiler(AbstractCompiler):
    """Cuda code compiler for RTPs."""
    def __init__(self, e):
