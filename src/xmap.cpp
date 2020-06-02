#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "stats.hpp"
#include "xmap.hpp"

namespace edm
{

void _xmap(CrossMap &result, const Dataset &ds, const LUT &lut,
           const DevTargets &targets, uint32_t E, int32_t tau, int32_t Tp)
{
    const auto distances = lut.distances;
    const auto indices = lut.indices;

    size_t scratch_size = ScratchTimeSeries::shmem_size(ds.extent(0));

    Kokkos::parallel_for(
        "EDM::xmap::lookup",
        Kokkos::TeamPolicy<>(targets.size(), Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            uint32_t tj = targets(member.league_rank());

            ScratchTimeSeries scratch(member.team_scratch(0), ds.extent(0));

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, ds.extent(0)),
                                 [=](uint32_t i) { scratch(i) = ds(i, tj); });

            member.team_barrier();

            CorrcoefState state;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, distances.extent(0)),
                [=](uint32_t i, CorrcoefState &upd) {
                    float pred = 0.0f;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(member, E + 1),
                        [=](uint32_t &e, float &p) {
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

void xmap(CrossMap &result, const Dataset &ds, const TimeSeries &library,
          const std::vector<uint32_t> &edims, uint32_t E_max, int32_t tau,
          int32_t Tp)
{
    std::vector<LUT> luts;

    // Allocate kNN tables
    for (uint32_t E = 1; E <= E_max; E++) {
        luts.push_back(LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    LUT tmp_lut(ds.extent(0), ds.extent(0));

    // Compute kNN tables for all E
    for (uint32_t E = 1; E <= E_max; E++) {
        knn(library, library, luts[E - 1], tmp_lut, E, tau, Tp, E + 1);
        normalize_lut(luts[E - 1]);
    }

    // Group time series by their optimal embedding dimensions
    std::vector<std::vector<uint32_t>> groups(E_max);
    for (uint32_t i = 0; i < ds.extent(1); i++) {
        groups[edims[i] - 1].push_back(i);
    }

    // Perform lookups
    for (uint32_t E = 1; E <= E_max; E++) {
        if (!groups[E - 1].size()) {
            continue;
        }

        // TODO This needs to be done only once
        HostTargets h_targets(groups[E - 1].data(), groups[E - 1].size());
        DevTargets targets("targets", h_targets.size());
        Kokkos::deep_copy(targets, h_targets);

        _xmap(result, ds, luts[E - 1], targets, E, tau, Tp);
    }
}

} // namespace edm
