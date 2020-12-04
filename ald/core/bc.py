from jinja2 import Template
import cgen
from abc import abstractmethod, ABC
import numpy as np
import pycuda.gpuarray as gpuarray


class AbstractBC:
    pass

# only rectangular domain BCs are implemented here.

class AbstractNoFlux(AbstractBC):
    def __init__(self, displacement = False, displacement_var = "dx"):
        if not isinstance(displacement, bool):
            raise TypeError("not a boolean.")
        # whether or not to record the particle displacement due to collision with the wall
        self.displacement = displacement
        self.displacement_var = displacement_var


    def validator(self, orientation):
        # orientation, -1 for left&bottom, +1 for right *top
        if not orientation in [-1, 1]:
            raise ValueError("orientation should be -1/1")

    def cuda_code(self, loc, orientation, position_var):
        self.validator(orientation)
        # bc block
        # move to contact
        move = cgen.Assign("{}[tid]".format(position_var), "{}".format(loc))
        # hard wall displacement
        hard_wall = cgen.Assign(
            "{}[tid]".format(self.displacement_var),
            "{}-{}[tid]".format(loc, position_var),
        )

        if self.displacement:
            exprs = [hard_wall, move]
        else:
            exprs = [move]

        block = cgen.Block(exprs)
        # penetration condition.
        if orientation == -1:
            compare = "<"
        else:
            compare = ">"
        bc = cgen.If("{0}[tid]{1}{2}".format(position_var, compare, loc), block)
        return bc


class LeftNoFlux(AbstractNoFlux):
    def __init__(self, displacement=False, displacement_var = "dx1"):
        super().__init__(displacement, displacement_var)

    def cuda_code(self, box):
        return super().cuda_code(box.left, -1, "x")


class RightNoFlux(AbstractNoFlux):
    def __init__(self, displacement=False, displacement_var = "dx2"):
        super().__init__(displacement, displacement_var)

    def cuda_code(self, box):
        return super().cuda_code(box.right, 1, "x")


class BottomNoFlux(AbstractNoFlux):
    def __init__(self, displacement=False, displacement_var = "dy1"):
        super().__init__(displacement, displacement_var)

    def cuda_code(self, box):
        return super().cuda_code(box.bottom, -1, "y")


class TopNoFlux(AbstractNoFlux):
    def __init__(self, displacement=False, displacement_var = "dy2"):
        super().__init__(displacement, displacement_var)

    def cuda_code(self, box):
        return super().cuda_code(box.top, 1, "y")


class AbstractPBC(AbstractBC):
    def cuda_code(self, loc, orientation, position_var, L):
        if not orientation in [-1, 1]:
            raise ValueError("orientation should be -1/1")
        # bc block
        if orientation == -1:
            # left or bottom
            compare = "<"
            move = cgen.ExpressionStatement("{}[tid] += {}".format(position_var, L))
            #
            crossing = cgen.ExpressionStatement("pass{}[tid] += -1".format(position_var))
        else:
            # right or top
            compare = ">"
            move = cgen.ExpressionStatement("{}[tid] -= {}".format(position_var, L))
            #
            crossing = cgen.ExpressionStatement("pass{}[tid] += 1".format(position_var))
        exprs = [move, crossing]
        block = cgen.Block(exprs)
        # penetration condition.

        bc = cgen.If("{0}[tid]{1}{2}".format(position_var, compare, loc), block)
        return bc


class LeftPBC(AbstractPBC):
    def cuda_code(self, box):
        return super().cuda_code(box.left, -1, "x", box.Lx)


class RightPBC(AbstractPBC):
    def cuda_code(self, box):
        return super().cuda_code(box.right, 1, "x", box.Lx)


class BottomPBC(AbstractPBC):
    def cuda_code(self, box):
        return super().cuda_code(box.bottom, -1, "y", box.Ly)


class TopPBC(AbstractPBC):
    def cuda_code(self, box):
        return super().cuda_code(box.top, 1, "y", box.Ly)








