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
        if isinstance(self.left, NoFlux):
            self.left_bc = "x[tid] = -{} / 2.0;".format(self.Lx)
        elif isinstance(self.left, Periodic):
            self.left_bc = "x[tid] += {};".format(self.Lx)
        else:
            raise NotImplementedError()

        if isinstance(self.right, NoFlux):
            self.right_bc = "x[tid] = {} / 2.0;".format(self.Lx)
        elif isinstance(self.right, Periodic):
            self.right_bc = "x[tid] -= {};".format(self.Lx)
        else:
            raise NotImplementedError()

        if isinstance(self.bottom, NoFlux):
            self.bottom_bc = "y[tid] = -{} / 2.0;".format(self.Ly)
        elif isinstance(self.bottom, Periodic):
            self.bottom_bc = "y[tid] += {};".format(self.Ly)
        else:
            raise NotImplementedError()

        if isinstance(self.top, NoFlux):
            self.top_bc = "y[tid] = {} / 2.0;".format(self.Ly)
        elif isinstance(self.top, Periodic):
            self.top_bc = "y[tid] -= {};".format(self.Ly)
        else:
            raise NotImplementedError()
