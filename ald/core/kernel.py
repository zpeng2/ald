from abc import ABC, abstractmethod



class AbstractKernel(ABC):
    """Kernel code is stored in kernel_code attribute"""
    def __init__(self, arg_list):
        """additional args needed to add to the kernel."""
        self.arg_list = arg_list
    @abstractmethod
    def generate_cuda_code(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass
