import kedm
import numpy as np
import pytest


# thetas = [0.0, 0.01, 0.1, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# df = pd.DataFrame()
# for theta in thetas:
#     res = pyEDM.SMap(dataFile="test/logistic_map.csv", columns="x",
#                      lib="1 200", pred="1 200", E=2, tau=-1, Tp=1,
#                      theta=theta)
#     series = res["predictions"]["Predictions"]
#     df = pd.concat([df, series.dropna().rename(f"theta={theta}")], axis=1)
@pytest.mark.parametrize("i", range(15))
def test_smap_varying_theta(pytestconfig, i):
    theta = [0.0, 0.01, 0.1, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9][i]

    ts = np.loadtxt(pytestconfig.rootdir / "test/logistic_map.csv",
                    delimiter=",", skiprows=1)[:, 1]
    valid = np.loadtxt(pytestconfig.rootdir / "test/smap_valid_logistic_map.csv",
                       delimiter=",", skiprows=1)

    lib = pred = ts
    prediction = kedm.smap(lib, pred, E=2, tau=1, Tp=1, theta=theta)

    assert prediction == pytest.approx(valid[:, i+1], abs=5e-2)


# Case 1:
# pyEDM.SMap(dataFrame=pyEDM.sampleData["block_3sp"], columns="x_t",
#            lib="1 100", pred="1 99", E=3, tau=-1, Tp=1, theta=1.0)
#
# Case 2:
#  pyEDM.SMap(dataFrame=pyEDM.sampleData["block_3sp"], columns="x_t",
#             lib="1 100", pred="101 194", E=3, tau=-1, Tp=1, theta=1.0)
#
# Case 3:
# pyEDM.SMap(dataFrame=pyEDM.sampleData["Lorenz5D"], columns="V1",
#            lib="1 600", pred="601 995", E=5, tau=-5, Tp=5, theta=1.0)
@pytest.mark.parametrize(
    "data_csv,valid_csv,lib_range,pred_range,E,tau,Tp,columns,target",
    [("block_3sp.csv", "smap_valid_blk3sp_x_1_100.csv",
      [0, 100], [0, 99], 3, 1, 1, 1, 1),
     ("block_3sp.csv", "smap_valid_blk3sp_x_1_100_101_195.csv",
      [0, 100], [98, 194], 3, 1, 1, 1, 1),
     ("LorenzData1000.csv", "smap_valid_Lorenz5D_V1_1_600_601_1000.csv",
      [0, 600], [580, 995], 5, 5, 5, 1, 1)])
def test_smap_values(pytestconfig, data_csv, valid_csv, lib_range,
                     pred_range, E, tau, Tp, columns, target):
    data = np.loadtxt(pytestconfig.rootdir / "test/" / data_csv,
                      skiprows=1, delimiter=",")
    valid = np.loadtxt(pytestconfig.rootdir / "test/" / valid_csv,
                       skiprows=1, delimiter=",")[:, 2]  # Predictions

    lib = data[lib_range[0]:lib_range[1], columns]
    pred = data[pred_range[0]:pred_range[1], columns]
    target = data[pred_range[0]:pred_range[1], target]

    prediction = kedm.smap(lib=lib, pred=pred, target=target, E=E,
                           tau=tau, Tp=Tp, theta=1.0)

    assert prediction == pytest.approx(valid, abs=1e-2)


def test_invalid_args():
    lib = np.random.rand(10)
    pred = np.random.rand(10)

    with pytest.raises(ValueError, match=r"E must be greater than zero"):
        kedm.smap(lib, pred, E=-1)

    with pytest.raises(ValueError, match=r"tau must be greater than zero"):
        kedm.smap(lib, pred, E=2, tau=-1)

    with pytest.raises(ValueError, match=r"Tp must be greater or equal to zero"):
        kedm.smap(lib, pred, E=2, tau=1, Tp=-1)

    with pytest.raises(ValueError, match=r"lib size is too small"):
        kedm.smap(np.random.rand(1), pred, E=2)

    with pytest.raises(ValueError, match=r"pred size is too small"):
        kedm.smap(lib, np.random.rand(1), E=2)
