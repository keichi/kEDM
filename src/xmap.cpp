#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "xmap.hpp"

namespace edm
{

void xmap(const Dataset ds, const std::vector<int> &edims, CrossMap rho,
          int E_max)
{
    // const auto tau = 1;
    // const auto Tp = 0;

    // std::vector<LUT> luts;

    // for (auto E = 1; E <= E_max; E++) {
    // luts.push_back(LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    // }

    // for (auto i = 0; i < ds.extent(1); i++) {
    // const auto library = TimeSeries(ds, Kokkos::AUTO);

    // for (auto E = 1; E <= E_max; E++) {
    // knn.run(library, library, luts[E - 1], E, tau, Tp, E + 1);
    // normalize_lut(luts[E - 1]);
    // }

    // const auto E = edims[i];

    // TimeSeries prediction("prediction", target.size() - (E - 1) * tau);

    // const auto distances = lut.distances;
    // const auto indices = lut.indices;

    // parallel_for(
    // "lookup", Kokkos::TeamPolicy<>(targets.extent(0), Kokkos::AUTO),
    // KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
    // int i = member.league_rank();

    // Kokkos::parallel_for(
    // Kokkos::TeamThreadRange(member, distances.extent(0)),
    // [=](size_t j) {
    // auto pred = 0.0f;

    // for (auto e = 0; e < E; e++) {
    // pred +=
    // ds(indices(j, e), targets(i)) * indices(j, e);
    // }
    // });
    // });
    // }
}

} // namespace edm
