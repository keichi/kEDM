#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "simplex.hpp"

namespace edm
{

void simplex(MutableTimeSeries result, TimeSeries lib, TimeSeries pred,
             TimeSeries target, int E, int tau, int Tp)
{
    int n_partial = (E - 1) * tau;

    if (lib.extent(0) > target.extent(0)) {
        throw std::invalid_argument(
            "lib must have no more time steps than target");
    }

    TmpDistances tmp("tmp_distances", pred.extent(0), lib.extent(0));
    SimplexLUT lut(pred.extent(0) - n_partial, E + 1);

    knn(lib, pred, lut, tmp, E, tau, Tp, E + 1);
    normalize_lut(lut);
    lookup(result, target, lut);
}

void simplex(MutableTimeSeries result, Dataset lib, Dataset pred,
             TimeSeries target, int E, int tau, int Tp)
{
    int n_partial = (E - 1) * tau;

    if (lib.extent(0) > target.extent(0)) {
        throw std::invalid_argument(
            "lib must have no more time steps than target");
    }

    TmpDistances tmp("tmp_distances", pred.extent(0), lib.extent(0));
    SimplexLUT lut(pred.extent(0) - n_partial, E + 1);

    knn(lib, pred, lut, tmp, E, tau, Tp, E + 1);
    normalize_lut(lut);
    lookup(result, target, lut);
}

void lookup(MutableTimeSeries result, TimeSeries target, SimplexLUT lut)
{
    Kokkos::Profiling::pushRegion("EDM::simplex");

    const auto distances = lut.distances;
    const auto indices = lut.indices;

    Kokkos::parallel_for(
        "EDM::simplex::lookup", distances.extent(0), KOKKOS_LAMBDA(int i) {
            float p = 0.0f;

            for (size_t j = 0; j < distances.extent(1); j++) {
                p += target(indices(i, j)) * distances(i, j);
            }

            result(i) = p;
        });

    Kokkos::Profiling::popRegion();
}

} // namespace edm
