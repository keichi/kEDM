#include <cassert>

#include <Kokkos_Core.hpp>

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

#include "thirdparty/simd/simd.hpp"

#include "knn.hpp"
#include "types.hpp"

namespace edm
{

void calc_distances(TimeSeries library, TimeSeries target,
                    TmpDistances distances, int n_library, int n_target, int E,
                    int tau)
{
    using simd_t = simd::simd<float, simd::simd_abi::native>;

    Kokkos::parallel_for(
        "EDM::knn::calc_distances",
        Kokkos::TeamPolicy<>(n_target, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member,
                                        distances.extent(1) / simd_t::size()),
                [=](int j) {
                    simd_t dist = simd_t(0.0f);

                    for (int e = 0; e < E; e++) {
                        simd_t diff = simd_t(target(i + e * tau)) -
                                      simd_t(library.data() +
                                                 j * simd_t::size() + e * tau,
                                             simd::element_aligned_tag());
                        dist += diff * diff;
                    }

                    dist.copy_to(distances.data() + i * distances.extent(1) +
                                     j * simd_t::size(),
                                 simd::element_aligned_tag());
                });

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member,
                                        distances.extent(1) / simd_t::size() *
                                            simd_t::size(),
                                        distances.extent(1)),
                [=](int j) {
                    float dist = 0.0f;

                    for (int e = 0; e < E; e++) {
                        float diff = target(i + e * tau) - library(j + e * tau);
                        dist += diff * diff;
                    }

                    distances(i, j) = dist;
                });

            // Ignore degenerate neighbor
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                    if (target.data() + i == library.data() + j) {
                        distances(i, j) = FLT_MAX;
                    }
                });
        });
}

#ifdef KOKKOS_ENABLE_CUDA
void partial_sort(TmpDistances distances, LUT out, int n_library, int n_target,
                  int top_k, int shift)
{
    const int team_size = 32;

    const size_t scratch_size =
        ScratchDistances::shmem_size(team_size, top_k) +
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
                    scratch_idx(r, k) = j;
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
                    out.distances(i, j) = sqrt(min_dist);
                    out.indices(i, j) =
                        scratch_idx(min_rank, scratch_head(min_rank)) + shift;

                    scratch_head(min_rank) =
                        min(scratch_head(min_rank) + 1, top_k);
                }
            });
        });
}
#else
void partial_sort(TmpDistances distances, LUT out, int n_library, int n_target,
                  int top_k, int shift)
{
    using std::min;

    const size_t scratch_size =
        ScratchDistances::shmem_size(top_k) + ScratchIndices::shmem_size(top_k);

    // Partially sort each row
    Kokkos::parallel_for(
        "EDM::knn::partial_sort",
        Kokkos::TeamPolicy<>(n_target, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerThread(scratch_size)),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            // Scratch views to hold the top-k elements
            ScratchDistances scratch_dist(member.thread_scratch(0), top_k);
            ScratchIndices scratch_idx(member.thread_scratch(0), top_k);

            Kokkos::single(Kokkos::PerThread(member), [=]() {
                const int i = member.league_rank() * member.team_size() +
                              member.team_rank();

                if (i >= n_target) return;

                for (int j = 0; j < n_library; j++) {
                    const float cur_dist = distances(i, j);

                    // Skip elements larger than the current k-th smallest
                    // element
                    if (j >= top_k && cur_dist > scratch_dist(top_k - 1)) {
                        continue;
                    }

                    int k = 0;
                    // Shift elements until the insertion point is found
                    for (k = min(j, top_k - 1); k > 0; k--) {
                        if (scratch_dist(k - 1) <= cur_dist) {
                            break;
                        }

                        // Shift element
                        scratch_dist(k) = scratch_dist(k - 1);
                        scratch_idx(k) = scratch_idx(k - 1);
                    }

                    // Insert the new element
                    scratch_dist(k) = cur_dist;
                    scratch_idx(k) = j;
                }

                // Compute L2 norms from SSDs and shift indices
                for (int j = 0; j < top_k; j++) {
                    out.distances(i, j) = sqrt(scratch_dist(j));
                    out.indices(i, j) = scratch_idx(j) + shift;
                }
            });
        });
}
#endif

void knn(TimeSeries library, TimeSeries target, LUT out, TmpDistances tmp,
         int E, int tau, int Tp, int top_k)
{
    Kokkos::Profiling::pushRegion("EDM::knn");

    assert(E > 0 && tau > 0 && Tp >= 0 && top_k > 0);

    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    assert(n_library > 0 && n_target > 0);
    assert(tmp.extent(0) >= static_cast<size_t>(n_target) &&
           tmp.extent(1) >= static_cast<size_t>(n_library));
    assert(out.distances.extent(0) == static_cast<size_t>(n_target) &&
           out.distances.extent(1) == static_cast<size_t>(top_k));

    // Calculate all-to-all distances
    calc_distances(library, target, tmp, n_library, n_target, E, tau);

    // Sort the distance matrix
    partial_sort(tmp, out, n_library, n_target, top_k, shift);

    Kokkos::Profiling::popRegion();
}

void normalize_lut(LUT lut)
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
