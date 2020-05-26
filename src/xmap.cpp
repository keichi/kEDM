#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "stats.hpp"
#include "xmap.hpp"

namespace edm
{

void xmap(CrossMap &result, const Dataset &ds, const TimeSeries &library,
          const std::vector<int> &edims, int E_max, int tau, int Tp)
{
    std::vector<LUT> luts;

    // Allocate kNN tables
    for (auto E = 1; E <= E_max; E++) {
        luts.push_back(LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    LUT cache(ds.extent(0), ds.extent(0));
    NearestNeighbors knn(cache);

    // Compute kNN tables for all E
    for (auto E = 1; E <= E_max; E++) {
        knn.run(library, library, luts[E - 1], E, tau, Tp, E + 1);
        normalize_lut(luts[E - 1]);
    }

    // Group time series by their optimal embedding dimensions
    std::vector<std::vector<uint32_t>> groups(E_max);
    for (auto i = 0u; i < ds.extent(1); i++) {
        groups[edims[i] - 1].push_back(i);
    }

    // Perform lookups
    for (auto E = 1; E <= E_max; E++) {
        if (!groups[E - 1].size()) {
            continue;
        }

        // TODO This needs to be done only once
        HostTargets h_targets(groups[E - 1].data(), groups[E - 1].size());
        DevTargets targets("targets", h_targets.size());
        Kokkos::deep_copy(targets, h_targets);

        const auto distances = luts[E - 1].distances;
        const auto indices = luts[E - 1].indices;

        Kokkos::parallel_for(
            "lookup", Kokkos::TeamPolicy<>(targets.size(), Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
                int j = member.league_rank();

                CorrcoefState state;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, distances.extent(0)),
                    [=](int i, CorrcoefState &upd) {
                        auto pred = 0.0f;

                        for (auto e = 0; e < E + 1; e++) {
                            pred +=
                                ds(indices(i, e), targets(j)) * distances(i, e);
                        }

                        float actual = ds((E - 1) * tau + Tp + i, targets(j));

                        upd += CorrcoefState(pred, actual);
                    },
                    Kokkos::Sum<CorrcoefState>(state));

                Kokkos::single(Kokkos::PerTeam(member), [=]() {
                    result(targets(j)) =
                        state.xy_m2 / sqrt(state.x_m2 * state.y_m2);
                });
            });
    }
}

} // namespace edm
