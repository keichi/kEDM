import kedm
import numpy as np
import pytest


def test_xmap(pytestconfig):
    PREFIX = pytestconfig.rootdir / "test"

    ds = np.loadtxt(PREFIX / "xmap_all_to_all_test_input.csv", delimiter=",")

    E = [kedm.edim(ds[:, i]) for i in range(ds.shape[1])]

    E_valid = np.loadtxt(PREFIX / "xmap_all_to_all_test_validation_e.csv",
                         delimiter=",", dtype=int)

    assert np.all(E == E_valid)

    rho_valid = np.loadtxt(PREFIX / "xmap_all_to_all_test_validation_rho.csv",
                           delimiter=",")

    assert kedm.xmap(ds, E) == pytest.approx(rho_valid, abs=1e-5)
