import kedm
import numpy as np
import pytest


@pytest.mark.parametrize("i", range(15))
def test_smap(pytestconfig, i):
    E, tau, Tp = 2, 1, 1
    theta = [0.0, 0.01, 0.1, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9][i]

    ts = np.loadtxt(pytestconfig.rootdir / "test/logistic_map.csv", skiprows=1)
    rho_valid = np.loadtxt(pytestconfig.rootdir / "test/logistic_map_validation.csv",
                           delimiter=",", skiprows=1, usecols=1)

    lib = ts[:100]
    pred = ts[100:200]

    prediction = kedm.smap(lib, pred, pred, E, tau, Tp, theta)

    rho = np.corrcoef(prediction[:-1], pred[(E-1)*tau+Tp:])[0][1]

    assert rho == pytest.approx(rho_valid[i], abs=1e-2)

    rho = kedm.eval_smap(lib, pred, pred, E, tau, Tp, theta)

    assert rho == pytest.approx(rho_valid[i], abs=1e-2)


def test_invalid_args():
    lib = np.random.rand(10)
    pred = np.random.rand(10)

    with pytest.raises(ValueError, match=r"E must be greater than zero"):
        kedm.smap(lib, pred, lib, E=-1)

    with pytest.raises(ValueError, match=r"tau must be greater than zero"):
        kedm.smap(lib, pred, lib, E=2, tau=-1)

    with pytest.raises(ValueError, match=r"Tp must be greater or equal to zero"):
        kedm.smap(lib, pred, lib, E=2, tau=1, Tp=-1)

    with pytest.raises(ValueError, match=r"lib size is too small"):
        kedm.smap(np.random.rand(1), pred, lib, E=2)

    with pytest.raises(ValueError, match=r"pred size is too small"):
        kedm.smap(lib, np.random.rand(1), lib, E=2)
