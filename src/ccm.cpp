#include <cfloat>
#include <random>

#include "thirdparty/pcg/include/pcg_random.hpp"
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>
#include <boost/math/distributions/binomial.hpp>

#include "ccm.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace edm
{

void full_sort(SimplexLUT lut, int n_lib, int n_pred, int n_partial, int Tp)
{
    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                    lut.distances(i, j) = sqrt(lut.distances(i, j));
                    lut.indices(i, j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(lut.distances, i, Kokkos::ALL),
                Kokkos::subview(lut.indices, i, Kokkos::ALL));
        });
}

void full_sort_with_scratch(SimplexLUT lut, int n_lib, int n_pred,
                            int n_partial, int Tp)
{
    size_t shmem_size =
        ScratchDistances1D::shmem_size(lut.distances.extent(1)) +
        ScratchIndices1D::shmem_size(lut.indices.extent(1));

    Kokkos::parallel_for(
        "EDM::ccm::sort",
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            ScratchDistances1D scratch_distances(member.team_scratch(0),
                                                 lut.distances.extent(1));
            ScratchIndices1D scratch_indices(member.team_scratch(0),
                                             lut.indices.extent(1));

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                    scratch_distances(j) = sqrt(lut.distances(i, j));
                    scratch_indices(j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(member, scratch_distances,
                                                   scratch_indices);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](int j) {
                                     lut.distances(i, j) = scratch_distances(j);
                                     lut.indices(i, j) = scratch_indices(j);
                                 });
        });
}

void partial_sort(SimplexLUT lut, int k, int n_lib, int n_pred, int n_partial,
                  int Tp)
{
    Kokkos::parallel_for(
        "EDM::ccm::partial_sort", n_pred, KOKKOS_LAMBDA(int i) {
            std::partial_sort_copy(
                Counter<int>(0), Counter<int>(n_lib), &lut.indices(i, 0),
                &lut.indices(i, k), [&](int a, int b) {
                    return lut.distances(i, a) < lut.distances(i, b);
                });

            for (int j = 0; j < k; j++) {
                int idx = lut.indices(i, j);
                lut.distances(i, j) = sqrt(lut.distances(i, idx));
                lut.indices(i, j) = idx + n_partial + Tp;
            }

            for (int j = k; j < n_lib; j++) {
                lut.distances(i, j) = FLT_MAX;
            }
        });
}

std::vector<float> ccm(TimeSeries lib, TimeSeries target,
                       const std::vector<int> &lib_sizes, int sample, int E,
                       int tau, int Tp, int seed, float accuracy)
{
    Kokkos::Profiling::pushRegion("EDM::ccm");

    int n_partial = (E - 1) * tau;
    int n_lib = lib.extent(0) - n_partial - Tp;
    int n_pred = lib.extent(0) - n_partial;

    std::vector<float> rhos;

    if (lib_sizes.empty()) {
        return rhos;
    }

    TmpDistances tmp("tmp_distances", n_pred, n_lib);
    SimplexLUT full_lut(n_pred, n_lib);

    // Compute pairwise distance matrix
    calc_distances(lib, lib, tmp, n_lib, n_pred, E, tau);

    Kokkos::deep_copy(full_lut.distances, tmp);

    bool use_scratch =
#ifdef KOKKOS_ENABLE_CUDA
        ScratchDistances1D::shmem_size(full_lut.distances.extent(1)) +
            ScratchIndices1D::shmem_size(full_lut.indices.extent(1)) <
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO).scratch_size_max(0);
#else
        false;
#endif

    // (Partially) Sort each row of the distance matrix
    if (accuracy < 1.0f) {
        // Calculate the probability of a row to be sampled
        int min_lib_size =
            *std::min_element(lib_sizes.begin(), lib_sizes.end());
        float min_sampling_prob =
            static_cast<float>(min_lib_size) / lib.extent(0);

        // If we find the top-`k` neighbors and each of the top-k neighbors is
        // sampled at a probability of `min_sampling_prob`, we want to ensure
        // that at least E+1 neighbors are sampled with a probability of
        // `accuracy`.
        int k = std::ceil(boost::math::binomial::find_minimum_number_of_trials(
            E + 1, min_sampling_prob, 1.0f - accuracy));

        k = std::min(std::max(k, E + 1), n_lib);

        partial_sort(full_lut, k, n_lib, n_pred, n_partial, Tp);
    } else {
        if (use_scratch) {
            full_sort_with_scratch(full_lut, n_lib, n_pred, n_partial, Tp);
        } else {
            full_sort(full_lut, n_lib, n_pred, n_partial, Tp);
        }
    }

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

            if (lib_size >= lib.extent_int(0)) {
                mask_mirror.set();
            } else {
                // Random sampling without replacement (Floyd's algorithm)
                for (int i = lib.extent_int(0) - lib_size;
                     i < lib.extent_int(0); i++) {
                    int r = rng(i);
                    mask_mirror.set(mask_mirror.test(r) ? i : r);
                }
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
