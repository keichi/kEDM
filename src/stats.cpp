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

    return state.rho();
}

void corrcoef(CrossMap &rho, const Dataset &ds, const TimeSeries &x)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
    using std::sqrt;
#endif

    Kokkos::parallel_for(
        "EDM::stats::corrcef", Kokkos::TeamPolicy<>(ds.extent(1), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const uint32_t j = member.league_rank();
            CorrcoefState state;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, min(x.extent(0), ds.extent(0))),
                [=](uint32_t i, CorrcoefState &upd) {
                    upd += CorrcoefState(x(i), ds(i, j));
                },
                Kokkos::Sum<CorrcoefState>(state));

            rho(j) = state.rho();
        });
}

} // namespace edm
