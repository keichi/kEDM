#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "smap.hpp"
#include "types.hpp"

namespace py = pybind11;

int edim(py::array_t<float> ts_arr, int E_max, int tau, int Tp)
{
    if (ts_arr.ndim() != 1) {
        throw std::invalid_argument("Expected a 1D array");
    }

    auto ts = edm::MutableTimeSeries("ts", ts_arr.shape(0));
    auto mirror_ts = Kokkos::create_mirror_view(ts);

    for (auto i = 0; i < ts_arr.shape(0); i++) {
        mirror_ts(i) = *ts_arr.data(i);
    }

    Kokkos::deep_copy(ts, mirror_ts);

    return edm::edim(ts, E_max, tau, Tp);
}

py::array_t<float> simplex(py::array_t<float> library_arr,
                           py::array_t<float> target_arr, int E, int tau,
                           int Tp)
{
    if (library_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("Expected 1D arrays");
    }

    auto n_library = library_arr.shape(0);
    auto n_target = target_arr.shape(0);
    auto n_prediction = n_target - (E - 1) * tau;

    auto library = edm::MutableTimeSeries("library", n_library);
    auto target = edm::MutableTimeSeries("target", n_target);
    auto prediction = edm::MutableTimeSeries("prediction", n_prediction);

    auto mirror_library = Kokkos::create_mirror_view(library);
    auto mirror_target = Kokkos::create_mirror_view(target);
    auto mirror_prediction = Kokkos::create_mirror_view(prediction);

    for (auto i = 0; i < n_library; i++) {
        mirror_library(i) = *library_arr.data(i);
    }

    for (auto i = 0; i < n_target; i++) {
        mirror_target(i) = *target_arr.data(i);
    }

    Kokkos::deep_copy(library, mirror_library);
    Kokkos::deep_copy(target, mirror_target);

    edm::TmpDistances tmp("tmp_distances", n_target, n_library);
    edm::LUT lut(n_target, E + 1);

    edm::knn(library, target, lut, tmp, E, tau, Tp, E + 1);
    edm::normalize_lut(lut);
    edm::simplex(prediction, library, lut);

    Kokkos::deep_copy(mirror_prediction, prediction);

    py::array_t<float> prediction_arr({n_prediction});

    for (auto i = 0; i < n_prediction; i++) {
        *prediction_arr.mutable_data(i) = mirror_prediction(i);
    }

    return prediction_arr;
}

py::array_t<float> smap(py::array_t<float> library_arr,
                        py::array_t<float> target_arr, int E, int tau, int Tp,
                        float theta)
{
    if (library_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("Expected 1D arrays");
    }

    auto n_library = library_arr.shape(0);
    auto n_target = target_arr.shape(0);
    auto n_prediction = n_target - (E - 1) * tau;

    auto library = edm::MutableTimeSeries("library", n_library);
    auto target = edm::MutableTimeSeries("target", n_target);
    auto prediction = edm::MutableTimeSeries("prediction", n_prediction);

    auto mirror_library = Kokkos::create_mirror_view(library);
    auto mirror_target = Kokkos::create_mirror_view(target);
    auto mirror_prediction = Kokkos::create_mirror_view(prediction);

    for (auto i = 0; i < n_library; i++) {
        mirror_library(i) = *library_arr.data(i);
    }

    for (auto i = 0; i < n_target; i++) {
        mirror_target(i) = *target_arr.data(i);
    }

    Kokkos::deep_copy(library, mirror_library);
    Kokkos::deep_copy(target, mirror_target);

    edm::smap(prediction, library, target, E, tau, Tp, theta);

    Kokkos::deep_copy(mirror_prediction, prediction);

    py::array_t<float> prediction_arr({n_prediction});

    for (auto i = 0; i < n_prediction; i++) {
        *prediction_arr.mutable_data(i) = mirror_prediction(i);
    }

    return prediction_arr;
}

PYBIND11_MODULE(_kedm, m)
{
    m.doc() = "Python bindings to kEDM";

    m.def("edim", &edim,
          "Infer the optimal embedding dimension of a time series",
          py::arg("ts"), py::arg("E_max") = 20, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.def("simplex", &simplex,
          "Predict a time series from another using Simplex projection",
          py::arg("library"), py::arg("target"), py::arg("E") = 2,
          py::arg("tau") = 1, py::arg("Tp") = 1);

    m.def("smap", &smap, "Predict a time series from another using S-Map",
          py::arg("library"), py::arg("target"), py::arg("E") = 2,
          py::arg("tau") = 1, py::arg("Tp") = 1, py::arg("theta") = 1.0f);

    m.add_object("_cleanup", py::capsule([]() { Kokkos::finalize(); }));

    Kokkos::initialize();
}
