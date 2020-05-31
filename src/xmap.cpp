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
    for (int E = 1; E <= E_max; E++) {
        luts.push_back(LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    LUT cache(ds.extent(0), ds.extent(0));
    NearestNeighbors knn(cache);

    // Compute kNN tables for all E
    for (int E = 1; E <= E_max; E++) {
        knn.run(library, library, luts[E - 1], E, tau, Tp, E + 1);
        normalize_lut(luts[E - 1]);
    }

    // Group time series by their optimal embedding dimensions
    std::vector<std::vector<uint32_t>> groups(E_max);
    for (int i = 0u; i < ds.extent(1); i++) {
        groups[edims[i] - 1].push_back(i);
    }

    // Perform lookups
    for (int E = 1; E <= E_max; E++) {
        if (!groups[E - 1].size()) {
            continue;
        }

        // TODO This needs to be done only once
        HostTargets h_targets(groups[E - 1].data(), groups[E - 1].size());
        DevTargets targets("targets", h_targets.size());
        Kokkos::deep_copy(targets, h_targets);

        const auto distances = luts[E - 1].distances;
        const auto indices = luts[E - 1].indices;

        using ScratchTimeSeries =
            Kokkos::View<float *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>;

        size_t scratch_size = ScratchTimeSeries::shmem_size(ds.extent(0));

        Kokkos::parallel_for(
            "lookup",
            Kokkos::TeamPolicy<>(targets.size(), Kokkos::AUTO)
                .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
                int tj = targets(member.league_rank());

                ScratchTimeSeries scratch(member.team_scratch(0), ds.extent(0));

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, ds.extent(0)),
                    [=](int i) { scratch(i) = ds(i, tj); });

                member.team_barrier();

                CorrcoefState state;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, distances.extent(0)),
                    [=](int i, CorrcoefState &upd) {
                        float pred = 0.0f;

                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(member, E + 1),
                            [=](int &e, float &p) {
                                p += scratch(indices(i, e)) * distances(i, e);
                            },
                            pred);

                        float actual = scratch((E - 1) * tau + Tp + i);

                        upd += CorrcoefState(pred, actual);
                    },
                    Kokkos::Sum<CorrcoefState>(state));

                Kokkos::single(Kokkos::PerTeam(member), [=]() {
                    result(tj) = state.xy_m2 / sqrt(state.x_m2 * state.y_m2);
                });
            });
    }
}

} // namespace edm
