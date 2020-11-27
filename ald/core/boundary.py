from jinja2 import Template


class AbstractBC:
    pass


class NoFlux(AbstractBC):
    def __repr__(self):
        return "NoFlux()"


class Periodic(AbstractBC):
    def __repr__(self):
        return "Periodic()"


class AbstractBox:
    """Abstract Simulation box"""

    pass


rectangle_box_bc_code = Template(
    """
    // BC in x direction.
    if (x[tid] < -Lx / 2.0) {
      {{left_bc}}
      passx[tid] += -1;
    } else if (x[tid] > Lx / 2.0) {
      {{right_bc}}
      passx[tid] += 1;
    }
    // BC in y direction.
    if (y[tid] > Ly / 2.0) {
      {{ top_bc }}
      passy[tid] += 1;
    } else if (y[tid] < -Ly / 2.0) {
      {{ bottom_bc }}
      passy[tid] += -1;
    }"""
)


class Box(AbstractBox):
    """Square simulation domain."""

    def __init__(
        self,
        Lx=1.0,
        Ly=1.0,
        left=Periodic(),
        right=Periodic(),
        bottom=NoFlux(),
        top=NoFlux(),
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        # bc is set by calling _generate_bc_cuda
        self.bc = None
        # generate cuda code strings
        self._generate_bc_cuda()

    @classmethod
    # easy constructor for common domain
    def from_freespace(cls, Lx=1.0, Ly=1.0):
        """Free space, periodic conditions."""
        return cls(
            Lx=Lx,
            Ly=Ly,
            left=Periodic(),
            right=Periodic(),
            bottom=Periodic(),
            top=Periodic(),
        )

    @classmethod
    def from_channel(cls, Ly=1.0):
        """Planar channel geometry."""
        return cls(
            Lx=Ly,
            Ly=Ly,
            left=Periodic(),
            right=Periodic(),
            bottom=NoFlux(),
            top=NoFlux(),
        )

    def __repr__(self):
        return "Box(Lx={:.3f}, Ly={:.3f}, left={}, right={}, bottom={},top={})".format(
            self.Lx, self.Ly, self.left, self.right, self.bottom, self.top
        )

    def _generate_bc_cuda(self):
        """Generate the boundary condition code block"""
        # TODO: make a single BC code block, this makes the cuda kernel more general
        # because one can have abps inside a cavity, in which case the current
        # implementation does not work.
        if isinstance(self.left, NoFlux):
            left_bc = "x[tid] = -{} / 2.0;".format(self.Lx)
        elif isinstance(self.left, Periodic):
            left_bc = "x[tid] += {};".format(self.Lx)
        else:
            raise NotImplementedError()

        if isinstance(self.right, NoFlux):
            right_bc = "x[tid] = {} / 2.0;".format(self.Lx)
        elif isinstance(self.right, Periodic):
            right_bc = "x[tid] -= {};".format(self.Lx)
        else:
            raise NotImplementedError()

        if isinstance(self.bottom, NoFlux):
            bottom_bc = "y[tid] = -{} / 2.0;".format(self.Ly)
        elif isinstance(self.bottom, Periodic):
            bottom_bc = "y[tid] += {};".format(self.Ly)
        else:
            raise NotImplementedError()

        if isinstance(self.top, NoFlux):
            top_bc = "y[tid] = {} / 2.0;".format(self.Ly)
        elif isinstance(self.top, Periodic):
            top_bc = "y[tid] -= {};".format(self.Ly)
        else:
            raise NotImplementedError()
        # source code for boundary condition in a rectangular domain.
        self.bc = rectangle_box_bc_code.render(
            left_bc=left_bc, right_bc=right_bc, bottom_bc=bottom_bc, top_bc=top_bc
        )
