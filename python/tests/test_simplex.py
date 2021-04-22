import kedm
import numpy as np
import pytest

@pytest.mark.parametrize("E", range(2, 6))
def test_simplex(pytestconfig, E):
    tau, Tp = 1, 1

    ts = np.loadtxt(pytestconfig.rootdir / "test/simplex_test_data.csv",
                    skiprows=1)
    valid = np.loadtxt(pytestconfig.rootdir / f"test/simplex_test_validation_E{E}.csv",
                       skiprows=1)

    library = ts[:len(ts)//2]
    target = ts[len(ts)//2-(E-1)*tau:len(ts)-(E-1)*tau]

    prediction = kedm.simplex(library, target, E, tau, Tp)

    assert prediction == pytest.approx(valid, abs=1e-2)


@pytest.mark.parametrize("E", range(1, 21))
def test_simplex_rho(pytestconfig, E):
    tau, Tp = 1, 1

    ts = np.loadtxt(pytestconfig.rootdir / "test/TentMap_rEDM.csv",
                    delimiter=",", skiprows=1, usecols=1)
    valid = np.loadtxt(pytestconfig.rootdir / "test/TentMap_rEDM_validation.csv",
                       delimiter=",", skiprows=1, usecols=1)

    library = ts[0:100]
    target = ts[200 - (E - 1) * tau:500]
    prediction = kedm.simplex(library, target, E, tau, Tp)

    rho = np.corrcoef(prediction[:-1], target[(E-1)*tau+Tp:])[0][1]
    rho_valid = valid[E-1]

    assert rho == pytest.approx(rho_valid)
