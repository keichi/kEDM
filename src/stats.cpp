#include "stats.hpp"

namespace edm
{

float corrcoef(TimeSeries x, TimeSeries y)
{
    CorrcoefSimpleState state;

    Kokkos::parallel_reduce(
        "EDM::stats::corrcoef", Kokkos::min(x.size(), y.size()),
        KOKKOS_LAMBDA(int i, CorrcoefSimpleState &upd) {
            upd += CorrcoefSimpleState(x(i), y(i));
        },
        Kokkos::Sum<CorrcoefSimpleState>(state));

    return state.rho();
}

void corrcoef(CrossMap rho, Dataset ds, TimeSeries x)
{
    Kokkos::parallel_for(
        "EDM::stats::corrcoef",
        Kokkos::TeamPolicy<>(ds.extent(1), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const size_t j = member.league_rank();
            CorrcoefSimpleState state;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member,
                                        Kokkos::min(x.extent(0), ds.extent(0))),
                [=](size_t i, CorrcoefSimpleState &upd) {
                    upd += CorrcoefSimpleState(x(i), ds(i, j));
                },
                Kokkos::Sum<CorrcoefSimpleState>(state));

            rho(j) = state.rho();
        });
}

float mae(TimeSeries x, TimeSeries y)
{
    int n = Kokkos::min(x.size(), y.size());
    float sum;

    Kokkos::parallel_reduce(
        "EDM::stats::mae", n,
        KOKKOS_LAMBDA(int i, float &upd) { upd += Kokkos::abs(x(i) - y(i)); },
        sum);

    return sum / n;
}

float mse(TimeSeries x, TimeSeries y)
{
    int n = Kokkos::min(x.size(), y.size());
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
