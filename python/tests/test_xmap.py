import kedm
import numpy as np
import pytest


def test_xmap(pytestconfig):
    PREFIX = pytestconfig.rootdir / "test"

    ds = np.loadtxt(PREFIX / "xmap_all_to_all_test_input.csv", delimiter=",",
                    skiprows=1)

    E = [kedm.edim(ds[:, i]) for i in range(ds.shape[1])]

    E_valid = np.loadtxt(PREFIX / "xmap_all_to_all_test_validation_e.csv",
                         delimiter=",", dtype=int, skiprows=1)

    assert np.all(E == E_valid)

    rho_valid = np.loadtxt(PREFIX / "xmap_all_to_all_test_validation_rho.csv",
                           delimiter=",", skiprows=1)

    assert kedm.xmap(ds, E) == pytest.approx(rho_valid, abs=1e-5)


def test_invalid_args():
    with pytest.raises(ValueError, match=r"Expected a 2D array"):
        kedm.xmap(np.random.rand(100), np.random.randint(20, size=100))

    with pytest.raises(ValueError, match=r"Number of time series must match "
                       "the number of embedding dimensions"):
        kedm.xmap(np.random.rand(100, 100), np.random.randint(20, size=10))

    with pytest.raises(ValueError, match=r"All embedding dimensions must be "
                       "larger than zero"):
        kedm.xmap(np.random.rand(100, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, -1])
