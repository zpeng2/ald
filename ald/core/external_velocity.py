import h5py
class ExternalVelocity:
    """Class to specify external velocities such as flow etc."""

    def __init__(self, ux=None, uy=None, omega=None):
        # type checking, speeds need to be None or c code in string.
        self.string_or_none(ux)
        self.string_or_none(uy)
        self.string_or_none(omega)
        self.ux = ux
        self.uy = uy
        self.omega = omega

    @staticmethod
    def string_or_none(x):
        if x is not None and not isinstance(x, str):
            raise TypeError("invalid argument type.")


class ZeroVelocity(ExternalVelocity):
    """No external velocities."""

    def __init__(self):
        super().__init__(ux="0", uy="0", omega="0")


class Poiseuille(ExternalVelocity):
    """Specify Poiseuille flow."""

    def __init__(self, uf=1.0, H=1.0):
        self.uf = uf
        self.H = H
        ux = "{0}*(1.0-4.0*yold[tid]*yold[tid]/({1}*{1}))".format(uf, H)
        # no flow in y direction.
        uy = "0"
        omega = "4*{0}*yold[tid]/({1}*{1})".format(uf, H)
        super().__init__(ux, uy, omega)
    def save2h5(self, file):
        """Save velocities to H5 file as attributs."""
        with h5py.File(file, "r+") as f:
            f.attrs["uf"] = self.uf
            f.attrs["H"] = self.H
            f.attrs["flow_type"] = "Poiseuille"


class ConstantUx(ExternalVelocity):
    def __init__(self, U=1.):
        ux = "{0}".format(U)
        # no flow in y direction.
        uy = "0"
        omega = "0"
        super().__init__(ux, uy, omega)
