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

    const auto library = ts;
    const auto target = ts;

    TmpDistances tmp("tmp_distances", ts.size(), ts.size());

    for (int E = 1; E <= E_max; E++) {
        SimplexLUT lut(target.size() - (E - 1) * tau, E + 1);

        knn(library, target, lut, tmp, E, tau, Tp, E + 1);
        normalize_lut(lut);

        MutableTimeSeries prediction("prediction",
                                     target.size() - (E - 1) * tau);

        lookup(prediction, library, lut);

        const auto range =
            std::make_pair((E - 1) * tau + Tp, target.extent_int(0));
        rho[E - 1] = corrcoef(prediction, Kokkos::subview(target, range));
    }

    Kokkos::Profiling::popRegion();

    return std::max_element(rho.begin(), rho.end()) - rho.begin() + 1;
}

} // namespace edm
