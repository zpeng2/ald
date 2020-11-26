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
    def __init__(self, L=1.0, H=1.0):
        pass
