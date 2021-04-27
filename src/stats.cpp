#include "stats.hpp"

namespace edm
{

float corrcoef(TimeSeries x, TimeSeries y)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
    using std::sqrt;
#endif

    CorrcoefState state;

    Kokkos::parallel_reduce(
        "EDM::stats::corrcef", min(x.size(), y.size()),
        KOKKOS_LAMBDA(int i, CorrcoefState &upd) {
            upd += CorrcoefState(x(i), y(i));
        },
        Kokkos::Sum<CorrcoefState>(state));

    return state.rho();
}

void corrcoef(CrossMap rho, Dataset ds, TimeSeries x)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
    using std::sqrt;
#endif

    Kokkos::parallel_for(
        "EDM::stats::corrcef", Kokkos::TeamPolicy<>(ds.extent(1), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const int j = member.league_rank();
            CorrcoefState state;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, min(x.extent(0), ds.extent(0))),
                [=](int i, CorrcoefState &upd) {
                    upd += CorrcoefState(x(i), ds(i, j));
                },
                Kokkos::Sum<CorrcoefState>(state));

            rho(j) = state.rho();
        });
}

float mae(TimeSeries x, TimeSeries y)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::abs;
    using std::min;
#endif

    int n = min(x.size(), y.size());
    float sum;

    Kokkos::parallel_reduce(
        "EDM::stats::mae", n,
        KOKKOS_LAMBDA(int i, float &upd) { upd += abs(x(i) - y(i)); }, sum);

    return sum / n;
}

float mse(TimeSeries x, TimeSeries y)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
#endif

    int n = min(x.size(), y.size());
    float sum;

    Kokkos::parallel_reduce(
        "EDM::stats::mse", n,
        KOKKOS_LAMBDA(int i, float &upd) {
            upd += (x(i) - y(i)) * (x(i) - y(i));
        },
        sum);

    return sum / n;
}

} // namespace edm
