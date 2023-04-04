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


# CCM with full library should be identical to Simplex
@pytest.mark.parametrize("E", range(1, 20))
def test_ccm_with_full_lib(pytestconfig, E):
    tau, Tp = 1, 0

    dataset = np.loadtxt(pytestconfig.rootdir / "test/sardine_anchovy_sst.csv",
                         skiprows=1, delimiter=",")

    anchovy, np_sst = dataset[:, 1], dataset[:, 4]

    prediction = kedm.simplex(anchovy, anchovy, target=np_sst, E=E, tau=tau,
                              Tp=Tp)
    simplex_rho = np.corrcoef(prediction, np_sst[(E-1)*tau:])[0][1]

    ccm_rho = kedm.ccm(anchovy, np_sst, lib_sizes=[78], sample=1, E=E,
                       tau=tau, Tp=Tp)[0]

    assert simplex_rho == pytest.approx(ccm_rho, abs=1e-6)


def test_invalid_args():
    with pytest.raises(ValueError, match=r"lib and target must be 1D arrays"):
        kedm.ccm(np.random.rand(10, 10), np.random.rand(10))

    with pytest.raises(ValueError, match=r"All lib_sizes must be larger than zero"):
        kedm.ccm(np.random.rand(100), np.random.rand(100),
                 lib_sizes=[10, 20, -10, 30])

    with pytest.raises(ValueError, match=r"All lib_sizes must not exceed lib size"):
        kedm.ccm(np.random.rand(100), np.random.rand(100),
                 lib_sizes=[10, 20, 50, 100, 200])

    with pytest.raises(ValueError, match=r"sample must be larger than zero"):
        kedm.ccm(np.random.rand(100), np.random.rand(100),
                 lib_sizes=range(10, 50, 10), sample=0)
