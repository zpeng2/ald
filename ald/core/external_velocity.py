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


class EmptyVelocity(ExternalVelocity):
    """No external velocities."""

    def __init__(self):
        super.__init__(ux="0", uy="0", omega="0")


class Poiseuille(ExternalVelocity):
    """Specify Poiseuille flow."""

    def __init__(self, uf=1.0, H=1.0):
        ux = "{0}*(1.0-4.0*yold[tid]*yold[tid]/({1}*{1}))".format(uf, H)
        # no flow in y direction.
        uy = "0"
        omega = "4*{0}*yold[tid]/({1}*{1})".format(uf, H)
        super().__init__(ux, uy, omega)
