#include <Kokkos_Core.hpp>

#include "edim.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"

namespace edm
{

float corrcoef(const TimeSeries &x, const TimeSeries &y)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
    using std::sqrt;
#endif

    CorrcoefState state;

    Kokkos::parallel_reduce(
        min(x.size(), y.size()),
        KOKKOS_LAMBDA(int i, CorrcoefState &upd) {
            upd += CorrcoefState(x(i), y(i));
        },
        Kokkos::Sum<CorrcoefState>(state));

    return state.xy_m2 / sqrt(state.x_m2 * state.y_m2);
}

uint32_t edim(const TimeSeries &ts, int E_max, int tau, int Tp)
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

        rho[E - 1] = corrcoef(prediction, shifted_target);
    }

    return std::max_element(rho.begin(), rho.end()) - rho.begin() + 1;
}

} // namespace edm
