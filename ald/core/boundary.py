from jinja2 import Template



class AbstractDomain:
    """Abstract Simulation box"""

    pass


class Box(AbstractDomain):
    """Square simulation domain."""

    def __init__(
        self,
        left=-0.5,
        right=0.5,
        bottom=-0.5,
        top=0.5,
    ):
        if right < left:
            raise ValueError("right <left")
        if top < bottom:
            raise ValueError("top < bottom")
        self.Lx = right - left
        self.Ly = top - bottom
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def __repr__(self):
        return "[{}, {}]x[{}, {}]".format(self.left, self.right, self.bottom, self.top)




# class AbstractBC:
#     pass


# class NoFlux(AbstractBC):
#     def __repr__(self):
#         return "NoFlux()"


# class Periodic(AbstractBC):
#     def __repr__(self):
#         return "Periodic()"


# class AbstractDomain:
#     """Abstract Simulation box"""

#     pass


# rectangle_box_bc_code = Template(
#     """
#     // BC in x direction.
#     if (x[tid] < {{left}}) {
#       {{left_bc}}
#       passx[tid] += -1;
#     } else if (x[tid] > {{right}}) {
#       {{right_bc}}
#       passx[tid] += 1;
#     }
#     // BC in y direction.
#     if (y[tid] > {{top}}) {
#       {{ top_bc }}
#       passy[tid] += 1;
#     } else if (y[tid] < {{bottom}}) {
#       {{ bottom_bc }}
#       passy[tid] += -1;
#     }"""
# )


# class Box(AbstractDomain):
#     """Square simulation domain."""

#     def __init__(
#         self,
#         left=-0.5,
#         right=0.5,
#         bottom=-0.5,
#         top=0.5,
#         leftbc=Periodic(),
#         rightbc=Periodic(),
#         bottombc=NoFlux(),
#         topbc=NoFlux(),
#     ):
#         if right < left:
#             raise ValueError("right <left")
#         if top < bottom:
#             raise ValueError("top < bottom")
#         self.Lx = right - left
#         self.Ly = top - bottom
#         self.left = left
#         self.right = right
#         self.bottom = bottom
#         self.top = top
#         self.leftbc = leftbc
#         self.rightbc = rightbc
#         self.bottombc = bottombc
#         self.topbc = topbc
#         # bc is set by calling _generate_bc_cuda
#         self.bc = None
#         # generate cuda code strings
#         self._generate_bc_cuda()

#     @classmethod
#     # easy constructor for common domain
#     def from_freespace(cls, Lx=1.0, Ly=1.0):
#         """Free space, periodic conditions."""
#         return cls(
#             left=-Lx / 2,
#             right=Lx / 2,
#             bottom=-Ly / 2,
#             top=Ly / 2,
#             leftbc=Periodic(),
#             rightbc=Periodic(),
#             bottombc=Periodic(),
#             topbc=Periodic(),
#         )

#     @classmethod
#     def from_channel(cls, Ly=1.0):
#         """Planar channel geometry."""
#         return cls(
#             left=-Ly / 2,
#             right=Ly / 2,
#             bottom=-Ly / 2,
#             top=Ly / 2,
#             leftbc=Periodic(),
#             rightbc=Periodic(),
#             bottombc=NoFlux(),
#             topbc=NoFlux(),
#         )

#     def __repr__(self):
#         return "Box([{}, {}]x[{},{}], left={}, right={}, bottom={},top={})".format(
#             self.left,
#             self.right,
#             self.bottom,
#             self.top,
#             self.leftbc,
#             self.rightbc,
#             self.bottombc,
#             self.topbc,
#         )

#     def _generate_bc_cuda(self):
#         """Generate the boundary condition code block"""
#         # TODO: make a single BC code block, this makes the cuda kernel more general
#         # because one can have abps inside a cavity, in which case the current
#         # implementation does not work.
#         if isinstance(self.leftbc, NoFlux):
#             left_bc = "x[tid] = {};".format(self.left)
#         elif isinstance(self.leftbc, Periodic):
#             left_bc = "x[tid] += {};".format(self.Lx)
#         else:
#             raise NotImplementedError()

#         if isinstance(self.rightbc, NoFlux):
#             right_bc = "x[tid] = {};".format(self.right)
#         elif isinstance(self.rightbc, Periodic):
#             right_bc = "x[tid] -= {};".format(self.Lx)
#         else:
#             raise NotImplementedError()

#         if isinstance(self.bottombc, NoFlux):
#             bottom_bc = "y[tid] = {};".format(self.bottom)
#         elif isinstance(self.bottombc, Periodic):
#             bottom_bc = "y[tid] += {};".format(self.Ly)
#         else:
#             raise NotImplementedError()

#         if isinstance(self.topbc, NoFlux):
#             top_bc = "y[tid] = {};".format(self.top)
#         elif isinstance(self.topbc, Periodic):
#             top_bc = "y[tid] -= {};".format(self.Ly)
#         else:
#             raise NotImplementedError()
#         # source code for boundary condition in a rectangular domain.
#         self.bc = rectangle_box_bc_code.render(
#             left=self.left,
#             right=self.right,
#             bottom=self.bottom,
#             top=self.top,
#             left_bc=left_bc,
#             right_bc=right_bc,
#             bottom_bc=bottom_bc,
#             top_bc=top_bc,
#         )
