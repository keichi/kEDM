#include <cassert>

#include <Kokkos_Core.hpp>

#include "simplex.hpp"

namespace edm
{

void simplex(TimeSeries &prediction, const TimeSeries &target, const LUT &lut)
{
    const auto distances = lut.distances;
    const auto indices = lut.indices;

    assert(prediction.size() == distances.extent(0));

    Kokkos::parallel_for(
        "EDM::simplex::lookup", distances.extent(0), KOKKOS_LAMBDA(int i) {
            float pred = 0.0f;

            for (size_t j = 0; j < distances.extent(1); j++) {
                pred += target(indices(i, j)) * distances(i, j);
            }

            prediction(i) = pred;
        });
}

} // namespace edm
