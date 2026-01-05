#include <random>

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

template <class T> struct find_result {
    T val;
    bool found;

    KOKKOS_INLINE_FUNCTION find_result() : val(0), found(false) {}

    KOKKOS_INLINE_FUNCTION find_result &operator+=(const find_result &src)
    {
        if (src.found) {
            found = src.found;
            val = src.val;
        }
        return *this;
    }
};

} // namespace edm
namespace Kokkos
{
template <class T> struct reduction_identity<struct edm::find_result<T>> {
    KOKKOS_FORCEINLINE_FUNCTION static edm::find_result<T> sum()
    {
        return edm::find_result<T>();
    }
};
} // namespace Kokkos

namespace edm
{

void full_sort(SimplexLUT lut, int n_lib, int n_pred, int n_partial, int Tp)
{
    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](size_t j) {
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
                Kokkos::TeamThreadRange(member, n_lib), [=](size_t j) {
                    scratch_distances(j) = sqrt(lut.distances(i, j));
                    scratch_indices(j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(member, scratch_distances,
                                                   scratch_indices);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](size_t j) {
                                     lut.distances(i, j) = scratch_distances(j);
                                     lut.indices(i, j) = scratch_indices(j);
                                 });
        });
}

const unsigned int RADIX_BITS = 8;
const unsigned int RADIX_SIZE = 1 << RADIX_BITS;
const unsigned int RADIX_MASK = RADIX_SIZE - 1;

void partial_sort(SimplexLUT lut, int k, int n_lib, int n_pred, int n_partial,
                  int Tp)
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

            for (int digit_pos = 32 - RADIX_BITS; digit_pos >= 0 && !found;
                 digit_pos -= RADIX_BITS) {
                Kokkos::single(Kokkos::PerTeam(member), [=] {
                    for (unsigned int j = 0; j < RADIX_SIZE; j++) {
                        bins(j) = 0;
                    }
                });

                member.team_barrier();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_lib), [=](size_t j) {
                        unsigned int val = reinterpret_cast<unsigned int &>(
                            lut.distances(i, j));
                        if ((val & mask_desired) == desired) {
                            unsigned int digit = val >> digit_pos & RADIX_MASK;
                            Kokkos::atomic_inc(&bins(digit));
                        }
                    });

                member.team_barrier();

                for (unsigned int j = 0; j < RADIX_SIZE; j++) {
                    int count = bins(j);

                    if (count >= cur_k) {
                        mask_desired |= RADIX_MASK << digit_pos;
                        desired |= j << digit_pos;

                        if (count == 1) found = true;

                        break;
                    }

                    cur_k -= count;
                }

                // Needed because bins will be reset in the next iteration
                member.team_barrier();
            }

            find_result<float> res;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(member, n_lib),
                [=](size_t j, find_result<float> &upd) {
                    unsigned int val =
                        reinterpret_cast<unsigned int &>(lut.distances(i, j));
                    if ((val & mask_desired) == desired) {
                        upd.found = true;
                        upd.val = lut.distances(i, j);
                    }
                },
                Kokkos::Sum<find_result<float>>(res));

            Kokkos::parallel_scan(
                Kokkos::TeamThreadRange(member, n_lib),
                [=](size_t j, int &partial_sum, bool is_final) {
                    if (lut.distances(i, j) <= res.val) {
                        if (is_final && partial_sum < k) {
                            topk_dist(partial_sum) = sqrt(lut.distances(i, j));
                            topk_ind(partial_sum) = j + n_partial + Tp;
                        }
                        partial_sum++;
                    }
                });

            member.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, k),
                                 [=](size_t j) {
                                     lut.distances(i, j) = topk_dist(j);
                                     lut.indices(i, j) = topk_ind(j);
                                 });

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, k, n_lib),
                                 [=](size_t j) {
                                     lut.distances(i, j) = FLT_MAX;
                                     lut.indices(i, j) = -1;
                                 });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member,
                Kokkos::subview(lut.distances, i, Kokkos::make_pair(0, k)),
                Kokkos::subview(lut.indices, i, Kokkos::make_pair(0, k)));
        });
}

void partial_sort_bitonic(SimplexLUT lut, int k, int n_lib, int n_pred,
                          int n_partial, int Tp)
{
    typedef Kokkos::View<int *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        Scratch;

    typedef Kokkos::View<float *,
                         Kokkos::DefaultExecutionSpace::scratch_memory_space,
                         Kokkos::MemoryUnmanaged>
        ScratchDist;

    int lv0_scratch_size = Scratch::shmem_size(k) + ScratchDist::shmem_size(k);

    Kokkos::parallel_for(
        "EDM::ccm::partial_sort_bitonic",
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(lv0_scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Scratch topk_ind(member.team_scratch(0), k);
            ScratchDist topk_dist(member.team_scratch(0), k);

            // Initialize buffer with first k elements
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, k), [=](int j) {
                    if (j < n_lib) {
                        topk_dist(j) = lut.distances(i, j);
                        topk_ind(j) = j + n_partial + Tp;
                    } else {
                        topk_dist(j) = FLT_MAX;
                        topk_ind(j) = -1;
                    }
                });

            member.team_barrier();

            // Initial bitonic sort on k elements
            Kokkos::Experimental::sort_by_key_team(member, topk_dist, topk_ind);

            // Process remaining n-k elements
            for (int j = k; j < n_lib; j++) {
                float new_dist = lut.distances(i, j);

                // Only insert if new element is smaller than current max
                if (new_dist < topk_dist(k - 1)) {
                    // Find insertion position using binary search (single thread)
                    int insert_pos = k - 1;
                    Kokkos::single(Kokkos::PerTeam(member), [&]() {
                        int lo = 0, hi = k - 1;
                        while (lo < hi) {
                            int mid = (lo + hi) / 2;
                            if (topk_dist(mid) < new_dist) {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        insert_pos = lo;
                    });

                    member.team_barrier();

                    // Shift elements to the right (parallel)
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(member, insert_pos, k - 1),
                        [=](int idx) {
                            int src = k - 2 - (idx - insert_pos);
                            topk_dist(src + 1) = topk_dist(src);
                            topk_ind(src + 1) = topk_ind(src);
                        });

                    member.team_barrier();

                    // Insert new element (single thread)
                    Kokkos::single(Kokkos::PerTeam(member), [=]() {
                        topk_dist(insert_pos) = new_dist;
                        topk_ind(insert_pos) = j + n_partial + Tp;
                    });

                    member.team_barrier();
                }
            }

            // Write results with sqrt transformation
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, k), [=](int j) {
                    lut.distances(i, j) = sqrt(topk_dist(j));
                    lut.indices(i, j) = topk_ind(j);
                });

            // Fill remaining with sentinel values
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, k, n_lib), [=](int j) {
                    lut.distances(i, j) = FLT_MAX;
                    lut.indices(i, j) = -1;
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

                    for (size_t j = 0;
                         j < static_cast<size_t>(n_lib) && selected < E + 1;
                         j++) {
                        int idx = full_lut.indices(i, j);

                        // This means we ran out of (partially) sorted items
                        if (idx < 0) break;

                        if (mask.test(idx)) {
                            lut.distances(i, selected) =
                                full_lut.distances(i, j);
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

} // namespace edm
