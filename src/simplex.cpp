#include <Kokkos_Core.hpp>

#include "simplex.hpp"

namespace edm
{

void simplex(MutableTimeSeries prediction, TimeSeries target, LUT lut)
{
    Kokkos::Profiling::pushRegion("EDM::simplex");

    const auto distances = lut.distances;
    const auto indices = lut.indices;

    if (prediction.size() != distances.extent(0)) {
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

} // namespace edm
