#include <cassert>

#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "types.hpp"

namespace edm
{

void knn(const TimeSeries &library, const TimeSeries &target, LUT &out,
         LUT &tmp, int E, int tau, int Tp, int top_k)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::max;
    using std::min;
#endif

    Kokkos::Profiling::pushRegion("EDM::knn");

    assert(E > 0 && tau > 0 && Tp >= 0 && top_k > 0);

    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    assert(n_library > 0 && n_target > 0);

    const auto distances = tmp.distances;
    const auto indices = tmp.indices;

    assert(distances.extent(0) >= n_target && distances.extent(1) >= n_library);

    size_t scratch_size = ScratchTimeSeries::shmem_size(E);

#ifdef KOKKOS_ENABLE_CUDA
    // Calculate all-to-all distances
    Kokkos::parallel_for(
        "EDM::knn::calc_distances",
        Kokkos::TeamPolicy<>(n_target, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const int i = member.league_rank();

            ScratchTimeSeries scratch_target(member.team_scratch(0), E);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, E),
                [=](int e) { scratch_target(e) = target(i + e * tau); });

            member.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                    // Ignore degenerate neighbor
                    if (target.data() + i == library.data() + j) {
                        distances(i, j) = FLT_MAX;
                        return;
                    }

                    float dist = 0.0f;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(member, E),
                        [=](int e, float &d) {
                            float diff =
                                scratch_target(e) - library(j + e * tau);
                            d += diff * diff;
                        },
                        dist);

                    distances(i, j) = dist;
                });
        });
#else
    Kokkos::parallel_for(
        "EDM::knn::calc_distances",
        Kokkos::TeamPolicy<>(n_target, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i =
                member.league_rank() * member.team_size() + member.team_rank();

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n_library),
                                 [=](int j) { distances(i, j) = 0.0f; });

            for (int e = 0; e < E; e++) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, n_library), [=](int j) {
                        float diff = target(i + e * tau) - library(j + e * tau);

                        distances(i, j) += diff * diff;
                    });
            }

            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, n_library), [=](int j) {
                    // Ignore degenerate neighbor
                    if (target.data() + i == library.data() + j) {
                        distances(i, j) = FLT_MAX;
                    }
                });
        });
#endif

#ifdef KOKKOS_ENABLE_CUDA
    const int team_size = 32;
#else
    const int team_size = 1;
#endif

    scratch_size = ScratchDistances::shmem_size(team_size, top_k) +
                   ScratchIndices::shmem_size(team_size, top_k) +
                   Kokkos::View<int *, DevScratchSpace,
                                Kokkos::MemoryUnmanaged>::shmem_size(team_size);

    // Partially sort each row
    Kokkos::parallel_for(
        "EDM::knn::partial_sort",
        Kokkos::TeamPolicy<>(n_target, team_size)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            // Scratch views to hold the top-k elements
            ScratchDistances scratch_dist(member.team_scratch(0), team_size,
                                          top_k);
            ScratchIndices scratch_idx(member.team_scratch(0), team_size,
                                       top_k);
            Kokkos::View<int *, DevScratchSpace, Kokkos::MemoryUnmanaged>
                scratch_head(member.team_scratch(0), team_size);

            const int i = member.league_rank();
            const int r = member.team_rank();

            scratch_head(r) = 0;
            for (int j = 0; j < top_k; j++) {
                scratch_dist(r, j) = FLT_MAX;
            }

            member.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                    const float cur_dist = distances(i, j);
                    const int cur_idx = j;

                    // Skip elements larger than the current k-th smallest
                    // element
                    if (j / team_size >= top_k &&
                        cur_dist > scratch_dist(r, top_k - 1)) {
                        return;
                    }

                    int k = 0;
                    // Shift elements until the insertion point is found
                    for (k = min(j / team_size, top_k - 1); k > 0; k--) {
                        if (scratch_dist(r, k - 1) <= cur_dist) {
                            break;
                        }

                        // Shift element
                        scratch_dist(r, k) = scratch_dist(r, k - 1);
                        scratch_idx(r, k) = scratch_idx(r, k - 1);
                    }

                    // Insert the new element
                    scratch_dist(r, k) = cur_dist;
                    scratch_idx(r, k) = cur_idx;
                });

            member.team_barrier();

            Kokkos::single(Kokkos::PerTeam(member), [=]() {
                // Each thread owns its top-k elements. Now aggregate the
                // global top-k elements to rank zero.
                for (int j = 0; j < top_k; j++) {
                    float min_dist = FLT_MAX;
                    int min_rank = 0;

                    for (int r = 0; r < team_size; r++) {
                        if (scratch_head(r) >= top_k) continue;

                        if (scratch_dist(r, scratch_head(r)) < min_dist) {
                            min_dist = scratch_dist(r, scratch_head(r));
                            min_rank = r;
                        }
                    }

                    // Compute L2 norms from SSDs and shift indices
                    distances(i, j) = sqrt(min_dist);
                    // indices(i, j) = scratch_idx(min_rank, 0) + shift;
                    indices(i, j) =
                        scratch_idx(min_rank, scratch_head(min_rank)) + shift;

                    scratch_head(min_rank) =
                        min(scratch_head(min_rank) + 1, top_k);
                }
            });
        });

    // Copy LUT from cache to output
    Kokkos::deep_copy(out.distances,
                      Kokkos::subview(distances, std::make_pair(0, n_target),
                                      std::make_pair(0, top_k)));
    Kokkos::deep_copy(out.indices,
                      Kokkos::subview(indices, std::make_pair(0, n_target),
                                      std::make_pair(0, top_k)));

    Kokkos::Profiling::popRegion();
} // namespace edm

void normalize_lut(LUT &lut)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::exp;
    using std::max;
    using std::min;
    using std::sqrt;
#endif

    Kokkos::Profiling::pushRegion("EDM::normalize_lut");

    auto distances = lut.distances;
    auto indices = lut.indices;
    const int L = distances.extent(0);
    const int top_k = distances.extent(1);

    // Normalize lookup table
    Kokkos::parallel_for(
        "EDM::normalize_distances", L, KOKKOS_LAMBDA(int i) {
            float sum_weights = 0.0f;
            float min_dist = FLT_MAX;
            float max_dist = 0.0f;

            for (int j = 0; j < top_k; j++) {
                float dist = distances(i, j);
                min_dist = min(min_dist, dist);
                max_dist = max(max_dist, dist);
            }

            for (int j = 0; j < top_k; j++) {
                const float dist = distances(i, j);

                float weighted_dist = 0.0f;

                if (min_dist > 0.0f) {
                    weighted_dist = exp(-dist / min_dist);
                } else {
                    weighted_dist = dist > 0.0f ? 0.0f : 1.0f;
                }

                const float weight = max(weighted_dist, MIN_WEIGHT);

                distances(i, j) = weight;

                sum_weights += weight;
            }

            for (int j = 0; j < top_k; j++) {
                distances(i, j) /= sum_weights;
            }
        });

    Kokkos::Profiling::popRegion();
}

} // namespace edm
