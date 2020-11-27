import ald
import numpy as np


def test_ABP():
    U0 = 0.1
    DT = 1.0
    tauR = 2.0
    particle = ald.ABP(U0=U0, tauR=tauR, DT=DT)
    assert particle.U0 == U0
    assert particle.DT == DT
    assert particle.tauR == tauR
    assert particle.DR == 1 / tauR


def test_RTP():
    particle = ald.RTP(U0=1.0, tauR=2.0)
    assert particle.U0 == 1.0
    assert particle.tauR == 2.0


def test_Pareto():
    particle = ald.Pareto(U0=1.0, tauR=1.0, alpha=1.5)
    assert np.isclose(
        particle.taum, (particle.alpha - 1) / particle.alpha * particle.tauR
    )
