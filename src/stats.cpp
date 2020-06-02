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
        "EDM::stats::corrcef", min(x.size(), y.size()),
        KOKKOS_LAMBDA(uint32_t i, CorrcoefState & upd) {
            upd += CorrcoefState(x(i), y(i));
        },
        Kokkos::Sum<CorrcoefState>(state));

    return state.xy_m2 / sqrt(state.x_m2 * state.y_m2);
}

} // namespace edm
