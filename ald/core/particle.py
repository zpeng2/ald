from abc import abstractmethod, ABC


class AbstractParticle:
    pass


class ABP(AbstractParticle):
    def __init__(self, U0=1.0, tauR=1.0, DT=1.0):
        self.U0 = U0
        self.tauR = tauR
        self.DR = 1 / tauR
        self.DT = DT

    def __repr__(self):
        return "ABP(U0 = {:.3f}, DR= {:.3f}, DT={:.3f})".format(
            self.U0,
            self.DR,
            self.DT,
        )


class AbstractRTP(AbstractParticle):
    def __init__(self, U0=1.0, tauR=1.0):
        # tauR is the mean runtime.
        self.tauR = tauR
        self.U0 = U0


class RTP(AbstractRTP):
    """Constant runtime RTP."""

    def __init__(self, U0=1.0, tauR=1.0):
        super().__init__(U0=U0, tauR=tauR)
        # cuda code for runtime.
        self.runtime_code = "{}".format(self.tauR)

    def __repr__(self):
        return "RTP(U0 = {:.3f}, tauR= {:.3f})".format(self.U0, self.tauR)


class Pareto(AbstractRTP):
    def __init__(self, U0=1.0, tauR=1.0, alpha=1.2):
        super().__init__(U0=U0, tauR=tauR)
        self.alpha = alpha
        self.taum = (alpha - 1) * tauR / alpha
        # see compiler for details.
        self.runtime_code = "pareto_runtime(&state[tid], {}, {})".format(
            self.tauR, self.alpha
        )

    def __repr__(self):
        return (
            "Pareto(U0 = {:.3f}, tauR= {:.3f}, alpha = {:.2f}, taum = {:.3f})".format(
                self.U0, self.tauR, self.alpha, self.taum
            )
        )
