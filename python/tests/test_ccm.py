import kedm
import numpy as np
import pytest


def test_ccm(pytestconfig):
    E, tau, Tp = 3, 1, 0
    lib_sizes = range(10, 80, 5)
    sample = 100

    dataset = np.loadtxt(pytestconfig.rootdir / "test/sardine_anchovy_sst.csv",
                         skiprows=1, delimiter=",")
    valid = np.loadtxt(pytestconfig.rootdir /
                       "test/anchovy_sst_ccm_validation.csv",
                       skiprows=1, delimiter=",")

    anchovy, np_sst = dataset[:, 1], dataset[:, 4]

    rhos1 = kedm.ccm(anchovy, np_sst, lib_sizes=lib_sizes, sample=sample, E=E,
                     tau=tau, Tp=Tp, seed=42)
    rhos2 = kedm.ccm(np_sst, anchovy, lib_sizes=lib_sizes, sample=sample, E=E, 
                     tau=tau, Tp=Tp, seed=42)

    assert rhos1 == pytest.approx(valid[:, 1])
    assert rhos2 == pytest.approx(valid[:, 2])


def test_invalid_args():
    with pytest.raises(ValueError, match=r"lib and target must be 1D arrays"):
        kedm.ccm(np.random.rand(10, 10), np.random.rand(10))

    with pytest.raises(ValueError, match=r"All lib_sizes must be larger than zero"):
        kedm.ccm(np.random.rand(100), np.random.rand(100),
                 lib_sizes=[10, 20, -10, 30])

    with pytest.raises(ValueError, match=r"sample must be larger than zero"):
        kedm.ccm(np.random.rand(100), np.random.rand(100),
                 lib_sizes=range(10, 50, 10), sample=0)
