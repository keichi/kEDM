#ifndef __EDIM_HPP__
#define __EDIM_HPP__

#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"
#include "types.hpp"

namespace edm
{

uint32_t edim(TimeSeries ts, int E_max, int tau, int Tp)
{
    std::vector<float> rho(E_max);

    const auto library = TimeSeries(ts, std::make_pair(0ul, ts.size() / 2));
    const auto target =
        TimeSeries(ts, std::make_pair(ts.size() / 2, ts.size()));

    LUT cache(ts.size(), ts.size());
    NearestNeighbors knn(cache);

    for (auto E = 1; E <= E_max; E++) {
        LUT lut(target.size() - (E - 1) * tau, E + 1);

        knn.run(library, target, lut, E, tau, Tp, E + 1);
        normalize_lut(lut);

        TimeSeries prediction("prediction", target.size() - (E - 1) * tau);
        TimeSeries shifted_target(
            target, std::make_pair((E - 1ul) * tau + Tp, target.size()));

        simplex(prediction, library, lut);

        Kokkos::fence();

        rho[E - 1] = corrcoef(prediction, shifted_target);
    }

    return std::max_element(rho.begin(), rho.end()) - rho.begin() + 1;
}

} // namespace edm

#endif
