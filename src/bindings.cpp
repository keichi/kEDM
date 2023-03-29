#include <Kokkos_Core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ccm.hpp"
#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "smap.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "xmap.hpp"

namespace py = pybind11;

void copy(edm::MutableTimeSeries dst, py::array_t<float> src)
{
    auto dst_mirror = Kokkos::create_mirror_view(dst);
    auto src_proxy = src.unchecked<1>();

    // This macro avoids a compiler warning on CUDA (calling a __host__
    // function from a __host__ __device__ function)
    KOKKOS_IF_ON_HOST(Kokkos::parallel_for(
                          "edm::bindings::copy",
                          Kokkos::RangePolicy<edm::HostSpace>(0, dst.extent(0)),
                          KOKKOS_LAMBDA(int i) { dst(i) = src_proxy(i); });)

    Kokkos::deep_copy(dst, dst_mirror);
}

void copy(edm::MutableDataset dst, py::array_t<float> src)
{
    auto dst_mirror = Kokkos::create_mirror_view(dst);
    auto src_proxy = src.unchecked<2>();

    KOKKOS_IF_ON_HOST(
        Kokkos::parallel_for(
            "edm::bindings::copy",
            Kokkos::MDRangePolicy<edm::HostSpace, Kokkos::Rank<2>>(
                {0, 0}, {dst.extent(0), dst.extent(1)}),
            KOKKOS_LAMBDA(int i, int j) {
                dst_mirror(i, j) = src_proxy(i, j);
            });)

    Kokkos::deep_copy(dst, dst_mirror);
}

void copy(py::array_t<float> dst, edm::TimeSeries src)
{
    auto dst_proxy = dst.mutable_unchecked<1>();
    auto src_mirror =
        Kokkos::create_mirror_view_and_copy(edm::HostSpace(), src);

    KOKKOS_IF_ON_HOST(
        Kokkos::parallel_for(
            "edm::bindings::copy",
            Kokkos::RangePolicy<edm::HostSpace>(0, src_mirror.extent(0)),
            KOKKOS_LAMBDA(int i) {
                // const_cast is required because dst_proxy is captured by
                // value.
                const_cast<float &>(dst_proxy(i)) = src_mirror(i);
            });)

    Kokkos::fence();
}

int edim(py::array_t<float> ts_arr, int E_max, int tau, int Tp)
{
    if (ts_arr.ndim() != 1) {
        throw std::invalid_argument("Expected a 1D array");
    }

    edm::MutableTimeSeries ts("ts", ts_arr.shape(0));

    copy(ts, ts_arr);

    return edm::edim(ts, E_max, tau, Tp);
}

py::array_t<float> simplex(py::array_t<float> lib_arr,
                           py::array_t<float> pred_arr,
                           py::array_t<float> target_arr, int E, int tau,
                           int Tp)
{
    if (lib_arr.ndim() != pred_arr.ndim()) {
        throw std::invalid_argument(
            "lib and pred must have same dimensionality");
    } else if (target_arr.ndim() > 1) {
        throw std::invalid_argument("target must be a 1D array");
    } else if (target_arr.ndim() == 0) {
        target_arr = lib_arr;
    }

    const auto n_lib = lib_arr.shape(0);
    const auto n_pred = pred_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_result = n_pred - (E - 1) * tau;

    if (lib_arr.ndim() == 1) {
        // Univariate prediction

        edm::MutableTimeSeries lib("lib", n_lib);
        edm::MutableTimeSeries pred("pred", n_pred);
        edm::MutableTimeSeries target("target", n_target);
        edm::MutableTimeSeries result("result", n_result);

        copy(lib, lib_arr);
        copy(pred, pred_arr);
        copy(target, target_arr);

        edm::simplex(result, lib, pred, target, E, tau, Tp);

        py::array_t<float> result_arr(n_result);

        copy(result_arr, result);

        return result_arr;
    } else if (lib_arr.ndim() == 2) {
        // Multivariate prediction

        const auto n_vars = lib_arr.shape(1);

        edm::MutableDataset lib("lib", n_lib, n_vars);
        edm::MutableDataset pred("pred", n_pred, n_vars);
        edm::MutableTimeSeries target("target", n_target);
        edm::MutableTimeSeries result("result", n_result);

        copy(lib, lib_arr);
        copy(pred, pred_arr);
        copy(target, target_arr);

        edm::simplex(result, lib, pred, target, E, tau, Tp);

        py::array_t<float> result_arr(n_result);

        copy(result_arr, result);

        return result_arr;
    } else {
        throw std::invalid_argument("lib and pred must be 1D or 2D arrays");
    }
}

float eval_simplex(py::array_t<float> lib_arr, py::array_t<float> pred_arr,
                   py::array_t<float> target_arr, int E, int tau, int Tp)
{
    if (lib_arr.ndim() != 1 || pred_arr.ndim() != 1) {
        throw std::invalid_argument("lib and pred must be 1D arrays");
    } else if (target_arr.ndim() > 1) {
        throw std::invalid_argument("target must be a 1D array");
    } else if (target_arr.ndim() == 0) {
        target_arr = lib_arr;
    }

    const auto n_lib = lib_arr.shape(0);
    const auto n_pred = pred_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_result = n_pred - (E - 1) * tau;

    edm::MutableTimeSeries lib("lib", n_lib);
    edm::MutableTimeSeries pred("pred", n_pred);
    edm::MutableTimeSeries target("target", n_target);
    edm::MutableTimeSeries result("result", n_result);

    copy(lib, lib_arr);
    copy(pred, pred_arr);
    copy(target, target_arr);

    edm::simplex(result, lib, pred, target, E, tau, Tp);

    const auto range = std::make_pair((E - 1) * tau + Tp, pred.extent_int(0));
    return edm::corrcoef(Kokkos::subview(pred, range), result);
}

py::array_t<float> smap(py::array_t<float> lib_arr, py::array_t<float> pred_arr,
                        py::array_t<float> target_arr, int E, int tau, int Tp,
                        float theta)
{
    if (lib_arr.ndim() != 1 || pred_arr.ndim() != 1) {
        throw std::invalid_argument("lib and pred must be 1D arrays");
    } else if (target_arr.ndim() > 1) {
        throw std::invalid_argument("target must be a 1D array");
    } else if (target_arr.ndim() == 0) {
        target_arr = lib_arr;
    }

    const auto n_lib = lib_arr.shape(0);
    const auto n_pred = pred_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_result = n_pred - (E - 1) * tau;

    edm::MutableTimeSeries lib("lib", n_lib);
    edm::MutableTimeSeries pred("pred", n_pred);
    edm::MutableTimeSeries target("target", n_target);
    edm::MutableTimeSeries result("result", n_result);

    copy(lib, lib_arr);
    copy(pred, pred_arr);
    copy(target, target_arr);

    edm::smap(result, lib, pred, target, E, tau, Tp, theta);

    py::array_t<float> result_arr(n_result);

    copy(result_arr, result);

    return result_arr;
}

float eval_smap(py::array_t<float> lib_arr, py::array_t<float> pred_arr,
                py::array_t<float> target_arr, int E, int tau, int Tp,
                float theta)
{
    if (lib_arr.ndim() != 1 || pred_arr.ndim() != 1) {
        throw std::invalid_argument("lib and pred must be 1D arrays");
    } else if (target_arr.ndim() > 1) {
        throw std::invalid_argument("target must be a 1D array");
    } else if (target_arr.ndim() == 0) {
        target_arr = lib_arr;
    }

    const auto n_lib = lib_arr.shape(0);
    const auto n_pred = pred_arr.shape(0);
    const auto n_target = target_arr.shape(0);
    const auto n_result = n_pred - (E - 1) * tau;

    edm::MutableTimeSeries lib("lib", n_lib);
    edm::MutableTimeSeries pred("pred", n_pred);
    edm::MutableTimeSeries target("target", n_target);
    edm::MutableTimeSeries result("result", n_result);

    copy(lib, lib_arr);
    copy(pred, pred_arr);
    copy(target, target_arr);

    edm::smap(result, lib, pred, target, E, tau, Tp, theta);

    const auto range = std::make_pair((E - 1) * tau + Tp, pred.extent_int(0));
    return edm::corrcoef(Kokkos::subview(pred, range), result);
}

std::vector<float> ccm(py::array_t<float> lib_arr,
                       py::array_t<float> target_arr,
                       const std::vector<int> &lib_sizes, int sample, int E,
                       int tau, int Tp, int seed)
{
    if (lib_arr.ndim() != 1 || target_arr.ndim() != 1) {
        throw std::invalid_argument("lib and target must be 1D arrays");
    } else if (*std::min_element(lib_sizes.begin(), lib_sizes.end()) <= 0) {
        throw std::invalid_argument("All lib_sizes must be larger than zero");
    } else if (sample <= 0) {
        throw std::invalid_argument("sample must be larger than zero");
    }

    edm::MutableTimeSeries lib("lib", lib_arr.shape(0));
    edm::MutableTimeSeries target("target", target_arr.shape(0));

    copy(lib, lib_arr);
    copy(target, target_arr);

    return edm::ccm(lib, target, lib_sizes, sample, E, tau, Tp, seed);
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

    copy(ds, ds_arr);

    std::vector<edm::SimplexLUT> luts;

    int E_max = *std::max_element(edims.begin(), edims.end());
    for (int E = 1; E <= E_max; E++) {
        luts.push_back(edm::SimplexLUT(ds.extent(0) - (E - 1) * tau, E + 1));
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
    m.doc() = R"pbdoc(
        Python bindings for kEDM
        ------------------------

        .. currentmodule:: _kedm

        .. autosummary::
           :toctree: _generate

           edim
           simplex
           eval_simplex
           smap
           eval_smap
           xmap
           get_kokkos_config
    )pbdoc";

    m.def("edim", &edim,
          R"doc(
          Infer the optimal embedding dimension of a time series.

          Args:
            timeseries: Time series
            E_max: Maximum embedding dimension (E is varied from 1 to E_max)
            tau: Time delay
            Tp: Prediction interval
          Returns:
            Optimal embedding dimension of the time series
          )doc",
          py::arg("timeseries"), py::arg("E_max") = 20, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.def("simplex", &simplex,
          R"doc(
          Predict a time series from another using Simplex projection.

          Args:
            lib: Library time series
            pred: Prediction time series
            target: Target time series
            E: Embedding dimension
            tau: Time delay
            Tp: Prediction interval
          Returns:
            Prediction result
          Note:
            If both lib and pred are 2D arrays, *mixed multivariate embedding*
            is peformed, where each time series is embedded into an
            E-dimensional state space.
          )doc",
          py::arg("lib"), py::arg("pred"), py::kw_only(),
          py::arg("target") = nullptr, py::arg("E") = 1, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.def("eval_simplex", &eval_simplex,
          R"doc(
          Predict a time series from another using Simplex projection and quantify its predictive skill.

          Args:
            lib: Library time series
            pred: Prediction time series
            target: Target time series
            E: Embedding dimension
            tau: Time delay
            Tp: Prediction interval
          Returns:
            Pearson's correlation coefficient between observation and prediction
          )doc",
          py::arg("lib"), py::arg("pred"), py::kw_only(),
          py::arg("target") = nullptr, py::arg("E") = 1, py::arg("tau") = 1,
          py::arg("Tp") = 1);

    m.def("smap", &smap,
          R"doc(
          Predict a time series from another using S-Map.

          Args:
            lib: Library time series
            pred: Prediction time series
            target: Target time series
            E: Embedding dimension
            tau: Time delay
            Tp: Prediction interval
            theta: Neighbor localization exponent
          Returns:
            Prediction result
          )doc",
          py::arg("lib"), py::arg("pred"), py::kw_only(),
          py::arg("target") = nullptr, py::arg("E") = 1, py::arg("tau") = 1,
          py::arg("Tp") = 1, py::arg("theta") = 1.0f);

    m.def("eval_smap", &eval_smap,
          R"doc(
          Predict a time series from another using S-Map and quantify its predictive skill.

          Args:
            lib: Library time series
            pred: Prediction time series
            target: Target time series
            E: Embedding dimension
            tau: Time delay
            Tp: Prediction interval
            theta: Neighbor localization exponent
          Returns:
            Pearson's correlation coefficient between observation and prediction
          )doc",
          py::arg("lib"), py::arg("pred"), py::kw_only(),
          py::arg("target") = nullptr, py::arg("E") = 1, py::arg("tau") = 1,
          py::arg("Tp") = 1, py::arg("theta") = 1.0f);

    m.def("ccm", &ccm,
          R"doc(
          Convergent Cross Mapping

          Args:
            lib: Library time series
            target: Target time series
            lib_sizes: Library sizes
            sample: Number of random samples
            E: Embedding dimension
            tau: Time delay
            Tp: Prediction interval
            seed: Random seed
          Returns:
            List of rhos
          )doc",
          py::arg("lib"), py::arg("target"), py::kw_only(),
          py::arg("lib_sizes") = std::vector<int>(), py::arg("sample") = 100,
          py::arg("E") = 1, py::arg("tau") = 1, py::arg("Tp") = 0,
          py::arg("seed") = 0);

    m.def("xmap", &xmap,
          R"doc(
          Infer the strength of causal interaction between multiple time
          series.

          Args:
            dataset: A 2D array where columns conrrespond to individual time series
            edims: Embedding dimension for each time series (can be computed using ``kedm.edim``)
            tau: Time delay
            Tp: Prediction interval
          Returns:
            A 2D array where each element represents the interaction strength
            between two time series.
          )doc",
          py::arg("dataset"), py::arg("edims"), py::arg("tau") = 1,
          py::arg("Tp") = 0);

    m.def("get_kokkos_config", &get_kokkos_config,
          R"doc(
          Get configuration of Kokkos that kEDM was built with.

          Returns:
            Kokkos configuration
          )doc");

    m.add_object("_cleanup", py::capsule([]() { Kokkos::finalize(); }));

    Kokkos::initialize();
}
