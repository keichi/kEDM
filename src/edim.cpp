#include <Kokkos_Core.hpp>

#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"

namespace edm
{

uint32_t edim(const TimeSeries &ts, uint32_t E_max, int32_t tau, int32_t Tp)
{
    std::vector<float> rho(E_max);

    auto library = TimeSeries(ts, std::make_pair(0ul, ts.size() / 2));
    auto target = TimeSeries(ts, std::make_pair(ts.size() / 2, ts.size()));

    LUT tmp_lut(ts.size(), ts.size());

    for (uint32_t E = 1; E <= E_max; E++) {
        LUT lut(target.size() - (E - 1) * tau, E + 1);

        knn(library, target, lut, tmp_lut, E, tau, Tp, E + 1);
        normalize_lut(lut);

        TimeSeries prediction("prediction", target.size() - (E - 1) * tau);
        TimeSeries shifted_target(
            target, std::make_pair((E - 1ul) * tau + Tp, target.size()));

        simplex(prediction, library, lut);

        rho[E - 1] = corrcoef(prediction, shifted_target);
    }

    return std::max_element(rho.begin(), rho.end()) - rho.begin() + 1;
}

} // namespace edm
