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

    lib = ts[:len(ts)//2]
    pred = ts[len(ts)//2-(E-1)*tau:len(ts)-(E-1)*tau]

    prediction = kedm.simplex(lib, pred, E=E, tau=tau, Tp=Tp)

    assert prediction == pytest.approx(valid, abs=1e-2)


def test_multivariate_simplex(pytestconfig):
    E, tau, Tp = 3, 1, 1

    data = np.loadtxt(pytestconfig.rootdir / "test/block_3sp.csv", skiprows=1,
                      delimiter=",")
    valid = np.loadtxt(pytestconfig.rootdir / "test/block_3sp_validation.csv",
                       skiprows=1, delimiter=",")

    # Columns #1, #4 and #7 are x_t, y_t and z_t
    lib = data[:99, [1, 4, 7]]
    pred = data[97:198, [1, 4, 7]]

    prediction = kedm.simplex(lib, pred, target=lib[:, 0], E=E, tau=tau, Tp=Tp)

    assert prediction == pytest.approx(valid, abs=1e-6)


# Columns #1, #4 and #7 are x_t, y_t and z_t
@pytest.mark.parametrize("valid_csv,lib_range,pred_range,columns,target",
                         [("block_3sp_valid_MV_xy_xy_lib_1_100_pred_101_195.csv",
                           [0, 100], [100, 195], [1, 4], 1),
                          ("block_3sp_valid_MV_xy_xy_lib_1_100_pred_1_100.csv",
                           [0, 100], [0,   100], [1, 4], 1),
                          ("block_3sp_valid_MV_xy_xx_lib_1_100_pred_101_195.csv",
                           [0, 100], [100, 195], [1, 4], 1),
                          ("block_3sp_valid_MV_xy_zy_lib_1_100_pred_101_195.csv",
                           [0, 100], [100, 195], [1, 4], 7)])
def test_MV_simplex(pytestconfig, valid_csv, lib_range, pred_range, columns, target):
    E, tau, Tp = 3, 1, 1

    data = np.loadtxt(pytestconfig.rootdir / "test/block_3sp.csv", skiprows=1,
                      delimiter=",")
    valid = np.loadtxt(pytestconfig.rootdir / "test/" / valid_csv,
                       skiprows=1, delimiter=",")[:, 2]  # Predictions

    lib = data[lib_range[0]:lib_range[1], columns]
    pred = data[pred_range[0]:pred_range[1], columns]
    target = data[lib_range[0]:lib_range[1], target]

    prediction = kedm.simplex(lib, pred, target=target, E=E, tau=tau, Tp=Tp)

    assert prediction == pytest.approx(valid, abs=1e-5)


def test_full_lib_simplex(pytestconfig):
    E, tau, Tp = 2, 1, 1

    ts = np.loadtxt(pytestconfig.rootdir / "test/simplex_test_data.csv",
                    skiprows=1)
    valid = np.loadtxt(pytestconfig.rootdir / f"test/simplex_full_lib_test_validation.csv",
                       skiprows=1)

    prediction = kedm.simplex(ts, ts, E=E, tau=tau, Tp=Tp)

    assert prediction == pytest.approx(valid, abs=1e-5)


@pytest.mark.parametrize("E", range(1, 21))
def test_simplex_rho(pytestconfig, E):
    tau, Tp = 1, 1

    ts = np.loadtxt(pytestconfig.rootdir / "test/TentMap_rEDM.csv",
                    delimiter=",", skiprows=1, usecols=1)
    valid = np.loadtxt(pytestconfig.rootdir / "test/TentMap_rEDM_validation.csv",
                       delimiter=",", skiprows=1, usecols=1)

    lib = ts[0:100]
    pred = ts[200 - (E - 1) * tau:500]
    prediction = kedm.simplex(lib, pred, E=E, tau=tau, Tp=Tp)

    rho = np.corrcoef(prediction[:-1], pred[(E-1)*tau+Tp:])[0][1]
    rho_valid = valid[E-1]

    assert rho == pytest.approx(rho_valid, abs=1e-6)

    rho = kedm.eval_simplex(lib, pred, E=E, tau=tau, Tp=Tp)

    assert rho == pytest.approx(rho_valid, abs=1e-6)


def test_invalid_args():
    lib = np.random.rand(10)
    pred = np.random.rand(10)

    with pytest.raises(ValueError, match=r"E must be greater than zero"):
        kedm.simplex(lib, pred, E=-1)

    with pytest.raises(ValueError, match=r"tau must be greater than zero"):
        kedm.simplex(lib, pred, E=2, tau=-1)

    with pytest.raises(ValueError, match=r"Tp must be greater or equal to zero"):
        kedm.simplex(lib, pred, E=2, tau=1, Tp=-1)

    with pytest.raises(ValueError, match=r"lib size is too small"):
        kedm.simplex(np.random.rand(2), pred, E=3)

    with pytest.raises(ValueError, match=r"pred size is too small"):
        kedm.simplex(lib, np.random.rand(2), E=3)

    with pytest.raises(ValueError, match=r"lib and pred must have same"
                                         r" dimensionality"):
        kedm.simplex(np.random.rand(10), np.random.rand(10, 2))

    with pytest.raises(ValueError, match=r"lib and pred must be 1D or 2D"
                                         r" arrays"):
        kedm.simplex(np.random.rand(10, 2, 3), np.random.rand(10, 3, 4))
