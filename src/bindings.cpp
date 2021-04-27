#include <sstream>

#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "smap.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "xmap.hpp"

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
        throw std::invalid_argument("Expected a 1D array");
    }

    const auto n_library = library_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_prediction = n_target - (E - 1) * tau;

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

    edm::simplex(prediction, library, target, E, tau, Tp);

    Kokkos::deep_copy(mirror_prediction, prediction);

    py::array_t<float> prediction_arr({n_prediction});

    for (auto i = 0; i < n_prediction; i++) {
        *prediction_arr.mutable_data(i) = mirror_prediction(i);
    }

    return prediction_arr;
}

float simplex_eval(py::array_t<float> library_arr,
                   py::array_t<float> target_arr, int E, int tau, int Tp)
{
    if (library_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("Expected a 1D array");
    }

    const auto n_library = library_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_prediction = n_target - (E - 1) * tau;

    auto library = edm::MutableTimeSeries("library", n_library);
    auto target = edm::MutableTimeSeries("target", n_target);
    auto prediction = edm::MutableTimeSeries("prediction", n_prediction);

    auto mirror_library = Kokkos::create_mirror_view(library);
    auto mirror_target = Kokkos::create_mirror_view(target);

    for (auto i = 0; i < n_library; i++) {
        mirror_library(i) = *library_arr.data(i);
    }

    for (auto i = 0; i < n_target; i++) {
        mirror_target(i) = *target_arr.data(i);
    }

    Kokkos::deep_copy(library, mirror_library);
    Kokkos::deep_copy(target, mirror_target);

    edm::simplex(prediction, library, target, E, tau, Tp);

    const auto range = std::make_pair((E - 1) * tau + Tp, target.extent_int(0));
    return edm::corrcoef(Kokkos::subview(target, range), prediction);
}

py::array_t<float> smap(py::array_t<float> library_arr,
                        py::array_t<float> target_arr, int E, int tau, int Tp,
                        float theta)
{
    if (library_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("Expected a 1D array");
    }

    const auto n_library = library_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_prediction = n_target - (E - 1) * tau;

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

float smap_eval(py::array_t<float> library_arr, py::array_t<float> target_arr,
                int E, int tau, int Tp, float theta)
{
    if (library_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("Expected a 1D array");
    }

    const auto n_library = library_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_prediction = n_target - (E - 1) * tau;

    auto library = edm::MutableTimeSeries("library", n_library);
    auto target = edm::MutableTimeSeries("target", n_target);
    auto prediction = edm::MutableTimeSeries("prediction", n_prediction);

    auto mirror_library = Kokkos::create_mirror_view(library);
    auto mirror_target = Kokkos::create_mirror_view(target);

    for (auto i = 0; i < n_library; i++) {
        mirror_library(i) = *library_arr.data(i);
    }

    for (auto i = 0; i < n_target; i++) {
        mirror_target(i) = *target_arr.data(i);
    }

    Kokkos::deep_copy(library, mirror_library);
    Kokkos::deep_copy(target, mirror_target);

    edm::smap(prediction, library, target, E, tau, Tp, theta);

    const auto range = std::make_pair((E - 1) * tau + Tp, target.extent_int(0));
    return edm::corrcoef(Kokkos::subview(target, range), prediction);
}

py::array_t<float> xmap(py::array_t<float> ds_arr,
                        const std::vector<int> &edims, int tau, int Tp)
{
    if (ds_arr.ndim() != 2) {
        throw std::invalid_argument("Expected a 2D array");
    } else if (static_cast<py::size_t>(ds_arr.shape(1)) != edims.size()) {
        throw std::invalid_argument("Number of time series must match the "
                                    "number of embedding dimensions");
    } else if (*std::min_element(edims.begin(), edims.end()) <= 0) {
        throw std::invalid_argument("All embedding dimensions must be larger "
                                    "than zero");
    }

    auto ds = edm::MutableDataset("dataset", ds_arr.shape(0), ds_arr.shape(1));
    auto mirror_ds = Kokkos::create_mirror_view(ds);

    for (py::ssize_t i = 0; i < ds_arr.shape(0); i++) {
        for (py::ssize_t j = 0; j < ds_arr.shape(1); j++) {
            mirror_ds(i, j) = *ds_arr.data(i, j);
        }
    }

    Kokkos::deep_copy(ds, mirror_ds);

    std::vector<edm::LUT> luts;

    int E_max = *std::max_element(edims.begin(), edims.end());
    for (int E = 1; E <= E_max; E++) {
        luts.push_back(edm::LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    edm::TmpDistances tmp("tmp_distances", ds.extent(0), ds.extent(0));

    std::vector<edm::Targets> groups;
    edm::group_ts(groups, edims, E_max);

    edm::CrossMap ccm("ccm", ds.extent(1));
    auto ccm_mirror = Kokkos::create_mirror_view(ccm);

    py::array_t<float> ccm_arr({ds_arr.shape(1), ds_arr.shape(1)});

    for (size_t i = 0; i < ds.extent(1); i++) {
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(ccm, ds, library, groups, luts, tmp, E_max, tau, Tp);

        Kokkos::deep_copy(ccm_mirror, ccm);

        for (py::ssize_t j = 0; j < ds_arr.shape(1); j++) {
            *ccm_arr.mutable_data(i, j) = ccm_mirror(j);
        }
    }

    return ccm_arr;
}

std::string get_kokkos_config()
{
    std::stringstream ss;
    Kokkos::print_configuration(ss);
    return ss.str();
}

PYBIND11_MODULE(_kedm, m)
{
    m.doc() = "Python bindings for kEDM";

    m.def("edim", &edim,
          "Infer the optimal embedding dimension of a time series",
          py::arg("timeseries"), py::arg("E_max") = 20, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.def("simplex", &simplex,
          "Predict a time series from another using Simplex projection",
          py::arg("library"), py::arg("target"), py::arg("E") = 2,
          py::arg("tau") = 1, py::arg("Tp") = 1);

    m.def("simplex_eval", &simplex_eval,
          "Predict a time series from another using Simplex projection and "
          "quantify its predictive skill",
          py::arg("library"), py::arg("target"), py::arg("E") = 2,
          py::arg("tau") = 1, py::arg("Tp") = 1);

    m.def("smap", &smap, "Predict a time series from another using S-Map",
          py::arg("library"), py::arg("target"), py::arg("E") = 2,
          py::arg("tau") = 1, py::arg("Tp") = 1, py::arg("theta") = 1.0f);

    m.def("smap_eval", &smap_eval,
          "Predict a time series from another using S-Map and quantify its "
          "predictive skill",
          py::arg("library"), py::arg("target"), py::arg("E") = 1,
          py::arg("tau") = 1, py::arg("Tp") = 1, py::arg("theta") = 1.0f);

    m.def("xmap", &xmap, "All-to-all cross mapping", py::arg("dataset"),
          py::arg("edims"), py::arg("tau") = 1, py::arg("Tp") = 0);

    m.def("get_kokkos_config", &get_kokkos_config, "Get Kokkos configuration");

    m.add_object("_cleanup", py::capsule([]() { Kokkos::finalize(); }));

    Kokkos::initialize();
}
