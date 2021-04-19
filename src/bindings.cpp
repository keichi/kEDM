#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "edim.hpp"
#include "types.hpp"

namespace py = pybind11;

int edim(py::array_t<float> arr, int E_max, int tau, int Tp)
{
    if (arr.ndim() != 1) {
        throw std::runtime_error("Expected a 1D array!");
    }

    auto ts = edm::MutableTimeSeries("ts", arr.shape(0));
    auto mirror_ts = Kokkos::create_mirror_view(ts);

    for (auto i = 0; i < arr.shape(0); i++) {
        mirror_ts(i) = *arr.data(i);
    }

    Kokkos::deep_copy(ts, mirror_ts);

    return edm::edim(ts, E_max, tau, Tp);
}

PYBIND11_MODULE(_kedm, m)
{
    m.doc() = "Python bindings to kEDM";

    m.def("edim", &edim,
          "Infer the optimal embedding dimension of a time series",
          py::arg("ts"), py::arg("E_max") = 20, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.add_object("_cleanup", py::capsule([]() { Kokkos::finalize(); }));

    Kokkos::initialize();
}
