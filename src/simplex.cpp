#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "simplex.hpp"

namespace edm
{

void simplex(MutableTimeSeries prediction, TimeSeries library,
             TimeSeries target, int E, int tau, int Tp)
{
    TmpDistances tmp("tmp_distances", target.extent(0), library.extent(0));
    SimplexLUT lut(target.extent(0) - (E - 1) * tau, E + 1);

    knn(library, target, lut, tmp, E, tau, Tp, E + 1);
    normalize_lut(lut);
    lookup(prediction, library, lut);
}

void lookup(MutableTimeSeries prediction, TimeSeries target, SimplexLUT lut)
{
    Kokkos::Profiling::pushRegion("EDM::simplex");

    const auto distances = lut.distances;
    const auto indices = lut.indices;

    if (prediction.extent(0) != distances.extent(0)) {
        throw new std::invalid_argument(
            "prediction size and LUT row count must be equal");
    }

    Kokkos::parallel_for(
        "EDM::simplex::lookup", distances.extent(0), KOKKOS_LAMBDA(int i) {
            float pred = 0.0f;

            for (size_t j = 0; j < distances.extent(1); j++) {
                pred += target(indices(i, j)) * distances(i, j);
            }

            prediction(i) = pred;
        });

    Kokkos::Profiling::popRegion();
}

void simplex(MutableDataset prediction, Dataset library, Dataset target, int E,
             int tau, int Tp)
{
    TmpDistances tmp("tmp_distances", target.extent(0), library.extent(0));
    SimplexLUT lut(target.extent(0) - (E - 1) * tau, E + 1);

    knn(library, target, lut, tmp, E, tau, Tp, E + 1);
    normalize_lut(lut);
    lookup(prediction, library, lut);
}

void lookup(MutableDataset prediction, Dataset target, SimplexLUT lut)
{
    Kokkos::Profiling::pushRegion("EDM::simplex");

    const auto distances = lut.distances;
    const auto indices = lut.indices;

    if (prediction.extent(0) != distances.extent(0)) {
        throw new std::invalid_argument(
            "prediction time steps and LUT row count must be equal");
    }

    // Loop over time steps
    Kokkos::parallel_for(
        "EDM::simplex::lookup", distances.extent(0), KOKKOS_LAMBDA(int i) {
            // Loop over time series
            for (size_t k = 0; k < target.extent(1); k++) {
                prediction(i, k) = 0.0f;
            }

            // Loop over embedding dimensions
            for (size_t j = 0; j < distances.extent(1); j++) {
                float weight = distances(i, j);
                int index = indices(i, j);

                // Loop over time series
                for (size_t k = 0; k < target.extent(1); k++) {
                    prediction(i, k) += target(index, k) * weight;
                }
            }
        });

    Kokkos::Profiling::popRegion();
}

} // namespace edm
