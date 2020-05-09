#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include <Kokkos_Core.hpp>

namespace edm
{

void simplex(TimeSeries prediction, const TimeSeries target, const LUT &lut)
{
    const auto distances = lut.distances;
    const auto indices = lut.indices;

    assert(prediction.size() == distances.extent(0));

    Kokkos::parallel_for(
        "lookup", distances.extent(0), KOKKOS_LAMBDA(int i) {
            auto pred = 0.0f;

            for (auto j = 0u; j < distances.extent(1); j++) {
                pred += target(indices(i, j)) * distances(i, j);
            }

            prediction(i) = pred;
        });
}

} // namespace edm

#endif
