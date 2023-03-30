#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>

#include <random>
#include "thirdparty/pcg/include/pcg_random.hpp"

#include "ccm.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"
#include "types.hpp"

namespace edm
{

std::vector<float> ccm(TimeSeries lib, TimeSeries target,
                       const std::vector<int> &lib_sizes, int sample, int E,
                       int tau, int Tp, int seed)
{
    Kokkos::Profiling::pushRegion("EDM::ccm");

    int n_partial = (E - 1) * tau;
    int n_lib = lib.extent(0) - n_partial - Tp;
    int n_pred = lib.extent(0) - n_partial;

    std::vector<float> rhos;

    TmpDistances tmp("tmp_distances", n_pred, n_lib);
    SimplexLUT full_lut(n_pred, n_lib);

    // Compute pairwise distance matrix
    calc_distances(lib, lib, tmp, n_lib, n_pred, E, tau);

    Kokkos::deep_copy(full_lut.distances, tmp);

    // Sort each row of the distance matrix
    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                    full_lut.distances(i, j) = sqrt(full_lut.distances(i, j));
                    full_lut.indices(i, j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(full_lut.distances, i, Kokkos::ALL),
                Kokkos::subview(full_lut.indices, i, Kokkos::ALL));
        });

    pcg32 rng;

    if (seed == 0) {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        rng.seed(seed_source);
    } else {
        rng.seed(seed);
    }

    // Bit mask representing which library row is sampled
    Kokkos::Bitset<HostSpace> mask_mirror(lib.extent(0));
    Kokkos::Bitset<DevSpace> mask(lib.extent(0));

    SimplexLUT lut(n_pred, E + 1);

    MutableTimeSeries prediction("prediction", n_pred);

    for (int lib_size : lib_sizes) {

        float rho_sum = 0.0f;

        for (int trial = 0; trial < sample; trial++) {
            mask_mirror.clear();

            // Random sampling without replacement (Floyd's algorithm)
            for (int i = lib.extent_int(0) - lib_size; i < lib.extent_int(0);
                 i++) {
                int r = rng(i);
                mask_mirror.set(mask_mirror.test(r) ? i : r);
            }

            Kokkos::deep_copy(mask, mask_mirror);

            // Scan each row of the full LUT and check if the neighbor is
            // sampled. Collect neighbors into sampled LUT until E + 1
            // neighbors are found for each row.
            Kokkos::parallel_for(
                "EDM::ccm::sample", n_pred, KOKKOS_LAMBDA(int i) {
                    // Number of neighbors found so far
                    int selected = 0;

                    for (int j = 0; j < n_lib && selected < E + 1; j++) {
                        int idx = full_lut.indices(i, j);

                        if (mask.test(idx)) {
                            lut.distances(i, selected) =
                                full_lut.distances(i, j);
                            lut.indices(i, selected) = idx;
                            selected++;
                        }
                    }
                });

            // Normalize LUT and make prediction
            normalize_lut(lut);
            lookup(prediction, target, lut);

            float rho = corrcoef(
                Kokkos::subview(target, std::make_pair(n_partial + Tp,
                                                       target.extent_int(0))),
                prediction);

            rho_sum += rho;
        }

        rhos.push_back(rho_sum / sample);
    }

    Kokkos::Profiling::popRegion();

    return rhos;
}

} // namespace edm
