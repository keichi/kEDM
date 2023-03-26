#include <Kokkos_Core.hpp>

#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"

namespace edm
{

int edim(TimeSeries ts, int E_max, int tau, int Tp)
{
    Kokkos::Profiling::pushRegion("EDM::edim");

    if (E_max <= 0) {
        throw std::invalid_argument("E_max must be greater than zero");
    } else if (tau <= 0) {
        throw std::invalid_argument("tau must be greater than zero");
    } else if (Tp < 0) {
        throw std::invalid_argument("Tp must be greater or equal to zero");
    }

    std::vector<float> rho(E_max);

    const auto library =
        TimeSeries(ts, std::make_pair<size_t, size_t>(0, ts.size() / 2));
    const auto target =
        TimeSeries(ts, std::make_pair(ts.size() / 2, ts.size()));

    TmpDistances tmp("tmp_distances", ts.size(), ts.size());

    for (int E = 1; E <= E_max; E++) {
        SimplexLUT lut(target.size() - (E - 1) * tau, E + 1);

        knn(library, target, lut, tmp, E, tau, Tp, E + 1);
        normalize_lut(lut);

        MutableTimeSeries prediction("prediction",
                                     target.size() - (E - 1) * tau);
        TimeSeries shifted_target(
            target,
            std::make_pair<size_t, size_t>((E - 1) * tau + Tp, target.size()));

        lookup(prediction, library, lut);

        rho[E - 1] = corrcoef(prediction, shifted_target);
    }

    Kokkos::Profiling::popRegion();

    return std::max_element(rho.begin(), rho.end()) - rho.begin() + 1;
}

} // namespace edm
