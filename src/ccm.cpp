#include <random>
#include <unordered_set>

#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <pcg_random.hpp>

#include "ccm.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"
#include "types.hpp"

namespace edm
{

void full_sort(TmpDistances distances, TmpIndices indices, int n_lib,
               int n_pred, int n_partial, int Tp)
{
    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](size_t j) {
                                     distances(i, j) = sqrt(distances(i, j));
                                     indices(i, j) = j + n_partial + Tp;
                                 });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(distances, i, Kokkos::ALL),
                Kokkos::subview(indices, i, Kokkos::ALL));
        });
}

void full_sort_with_scratch(TmpDistances distances, TmpIndices indices,
                            int n_lib, int n_pred, int n_partial, int Tp)
{
    size_t shmem_size = ScratchDistances1D::shmem_size(distances.extent(1)) +
                        ScratchIndices1D::shmem_size(indices.extent(1));

    Kokkos::parallel_for(
        "EDM::ccm::sort",
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(shmem_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            ScratchDistances1D scratch_distances(member.team_scratch(0),
                                                 distances.extent(1));
            ScratchIndices1D scratch_indices(member.team_scratch(0),
                                             indices.extent(1));

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](size_t j) {
                    scratch_distances(j) = sqrt(distances(i, j));
                    scratch_indices(j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(member, scratch_distances,
                                                   scratch_indices);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](size_t j) {
                                     distances(i, j) = scratch_distances(j);
                                     indices(i, j) = scratch_indices(j);
                                 });
        });
}

const unsigned int RADIX_BITS = 8;
const unsigned int RADIX_SIZE = 1 << RADIX_BITS;
const unsigned int RADIX_MASK = RADIX_SIZE - 1;

void partial_sort(TmpDistances distances, TmpIndices indices, int k, int n_lib,
                  int n_pred, int n_partial, int Tp)
{
    typedef Kokkos::View<int *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        Scratch;

    typedef Kokkos::View<float *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        ScratchDist;

    int lv0_scratch_size = Scratch::shmem_size(RADIX_SIZE) +
                           Scratch::shmem_size(k) + ScratchDist::shmem_size(k);

    Kokkos::parallel_for(
        "EDM::ccm::partial_sort",
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(lv0_scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            int cur_k = k;
            unsigned int mask_desired = 0, desired = 0;
            bool found = false;

            Scratch bins(member.team_scratch(0), RADIX_SIZE);
            Scratch topk_ind(member.team_scratch(0), k);
            ScratchDist topk_dist(member.team_scratch(0), k);

            // Radix select: find the k-th smallest element by processing
            // 8 bits at a time from MSB to LSB
            for (int digit_pos = 32 - RADIX_BITS; digit_pos >= 0 && !found;
                 digit_pos -= RADIX_BITS) {
                // Reset histogram bins
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, RADIX_SIZE),
                    [=](size_t j) { bins(j) = 0; });

                member.team_barrier();

                // Build histogram for the current digit position, only
                // counting elements that match the prefix found so far
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_lib), [=](size_t j) {
                        unsigned int val =
                            reinterpret_cast<unsigned int &>(distances(i, j));
                        if ((val & mask_desired) == desired) {
                            unsigned int digit = val >> digit_pos & RADIX_MASK;
                            Kokkos::atomic_inc(&bins(digit));
                        }
                    });

                member.team_barrier();

                // Find the bin containing the k-th smallest element
                for (unsigned int j = 0; j < RADIX_SIZE; j++) {
                    int count = bins(j);

                    if (count >= cur_k) {
                        // k-th element is in this bin; update the desired
                        // prefix
                        mask_desired |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        if (count == 1) found = true;

                        break;
                    }

                    // k-th element is in a later bin; adjust k
                    cur_k -= count;
                }

                // Needed because bins will be reset in the next iteration
                member.team_barrier();
            }

            // Compute the k-th smallest distance value (pivot) by setting
            // all undetermined bits to 1
            unsigned int pivot_bits = desired | ~mask_desired;
            float pivot = reinterpret_cast<float &>(pivot_bits);

            // Collect top-k elements (those with distance <= pivot) into
            // scratch memory using parallel scan to assign positions
            Kokkos::parallel_scan(
                Kokkos::TeamThreadRange(member, n_lib),
                [=](size_t j, int &partial_sum, bool is_final) {
                    if (distances(i, j) <= pivot) {
                        if (is_final && partial_sum < k) {
                            topk_dist(partial_sum) = sqrt(distances(i, j));
                            topk_ind(partial_sum) = j + n_partial + Tp;
                        }
                        partial_sum++;
                    }
                });

            member.team_barrier();

            // Sort top-k elements in scratch memory
            Kokkos::Experimental::sort_by_key_team(member, topk_dist, topk_ind);

            // Copy sorted top-k to output and fill remaining with sentinel
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](size_t j) {
                                     if (j < static_cast<size_t>(k)) {
                                         distances(i, j) = topk_dist(j);
                                         indices(i, j) = topk_ind(j);
                                     } else {
                                         distances(i, j) = FLT_MAX;
                                         indices(i, j) = -1;
                                     }
                                 });
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
        Kokkos::Profiling::popRegion();
        return rhos;
    }

    TmpDistances tmp_dist("tmp_distances", n_pred, n_lib);
    TmpIndices tmp_ind("tmp_indices", n_pred, n_lib);

    // Compute pairwise distance matrix
    calc_distances(lib, lib, tmp_dist, n_lib, n_pred, E, tau);

    bool use_scratch =
#ifdef KOKKOS_ENABLE_CUDA
        ScratchDistances1D::shmem_size(tmp_dist.extent(1)) +
            ScratchIndices1D::shmem_size(tmp_ind.extent(1)) <
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

        partial_sort(tmp_dist, tmp_ind, k, n_lib, n_pred, n_partial, Tp);
    } else {
        if (use_scratch) {
            full_sort_with_scratch(tmp_dist, tmp_ind, n_lib, n_pred, n_partial,
                                   Tp);
        } else {
            full_sort(tmp_dist, tmp_ind, n_lib, n_pred, n_partial, Tp);
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

            // Scan each row of the sorted distances/indices and check if the
            // neighbor is sampled. Collect neighbors into sampled LUT until
            // E + 1 neighbors are found for each row.
            Kokkos::parallel_for(
                "EDM::ccm::sample", n_pred, KOKKOS_LAMBDA(int i) {
                    // Number of neighbors found so far
                    int selected = 0;

                    for (size_t j = 0;
                         j < static_cast<size_t>(n_lib) && selected < E + 1;
                         j++) {
                        int idx = tmp_ind(i, j);

                        // This means we ran out of (partially) sorted items
                        if (idx < 0) break;

                        if (mask.test(idx)) {
                            lut.distances(i, selected) = tmp_dist(i, j);
                            lut.indices(i, selected) = idx;
                            selected++;
                        }
                    }

                    // Fill the rest
                    for (int j = selected; j < E + 1; j++) {
                        lut.distances(i, j) = FLT_MAX;
                        lut.indices(i, j) = 0;
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

std::vector<float> ccm_naive(TimeSeries lib, TimeSeries target,
                             const std::vector<int> &lib_sizes, int sample,
                             int E, int tau, int Tp, int seed)
{
    Kokkos::Profiling::pushRegion("EDM::ccm_naive");

    const int n_partial = (E - 1) * tau;
    const int n_lib = lib.extent(0) - n_partial - Tp;
    const int n_pred = lib.extent(0) - n_partial;

    std::vector<float> rhos;

    if (lib_sizes.empty()) {
        Kokkos::Profiling::popRegion();
        return rhos;
    }

    pcg32 rng;
    if (seed == 0) {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        rng.seed(seed_source);
    } else {
        rng.seed(seed);
    }

    int max_lib_size = *std::max_element(lib_sizes.begin(), lib_sizes.end());

    TmpDistances tmp_dist("tmp_distances", n_pred, max_lib_size);

    SimplexLUT lut(n_pred, E + 1);
    MutableTimeSeries prediction("prediction", n_pred);

    for (int lib_size : lib_sizes) {
        float rho_sum = 0.0f;

        for (int trial = 0; trial < sample; trial++) {
            Kokkos::View<int *, DevSpace> sampled("sampled_indices", lib_size);
            auto sampled_mirror = Kokkos::create_mirror_view(sampled);

            std::unordered_set<int> selected;
            int count = 0;
            for (int i = n_lib - lib_size; i < n_lib; i++) {
                int r = i <= 0 ? 0 : rng(i);
                if (selected.find(r) != selected.end()) {
                    sampled_mirror(count++) = i;
                    selected.insert(i);
                } else {
                    sampled_mirror(count++) = r;
                    selected.insert(r);
                }
            }

            Kokkos::deep_copy(sampled, sampled_mirror);

            // Compute and sort distance matrix
            calc_distances_sampled(lib, tmp_dist, sampled, lib_size, n_pred, E,
                                   tau);
            partial_sort_sampled(tmp_dist, sampled, lut, lib_size, n_pred,
                                 E + 1, n_partial + Tp);

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
