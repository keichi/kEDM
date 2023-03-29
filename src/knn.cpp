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

void calc_distances(TimeSeries lib, TimeSeries pred, TmpDistances distances,
                    int n_lib, int n_pred, int E, int tau)
{
    using simd_t = simd::simd<float, simd::simd_abi::native>;

#ifdef USE_SCRATCH_MEMORY
    const size_t scratch_size = ScratchTimeSeries::shmem_size(E);
#endif

    Kokkos::parallel_for(
        "EDM::knn::calc_distances",
#ifdef USE_SCRATCH_MEMORY
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
#else
        Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
#endif
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const int i = member.league_rank();

#ifdef USE_SCRATCH_MEMORY
            // Load subset of pred time series to team scratch
            ScratchTimeSeries scratch_pred(member.team_scratch(0), E);

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, E),
                [=](int e) { scratch_pred(e) = pred(i + e * tau); });
#endif

            member.team_barrier();

#ifdef USE_SIMD_PRIMITIVES
            // Vectorized loop
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib / simd_t::size()),
                [=](int j) {
                    simd_t dist = simd_t(0.0f);

                    for (int e = 0; e < E; e++) {
#ifdef USE_SCRATCH_MEMORY
                        simd_t diff = simd_t(scratch_pred(e)) -
                                      simd_t(&lib(j * simd_t::size() + e * tau),
                                             simd::element_aligned_tag());
#else
                        simd_t diff = simd_t(pred(i + e * tau)) -
                                      simd_t(&lib(j * simd_t::size() + e * tau),
                                             simd::element_aligned_tag());
#endif
                        dist += diff * diff;
                    }

                    dist.copy_to(&distances(i, j * simd_t::size()),
                                 simd::element_aligned_tag());
                });
#endif

            // Remainder loop
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member,
#ifdef USE_SIMD_PRIMITIVES
                                        n_lib / simd_t::size() * simd_t::size(),
#else
                                        0,
#endif
                                        n_lib),
                [=](int j) {
                    float dist = 0.0f;

                    for (int e = 0; e < E; e++) {
#ifdef USE_SCRATCH_MEMORY
                        float diff = scratch_pred(e) - lib(j + e * tau);
#else
                        float diff = pred(i + e * tau) - lib(j + e * tau);
#endif
                        dist += diff * diff;
                    }

                    distances(i, j) = dist;
                });

            // Ignore degenerate neighbors
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](int j) {
                                     if (distances(i, j) == 0.0f) {
                                         distances(i, j) = FLT_MAX;
                                     }
                                 });
        });
}

void calc_distances(Dataset lib, Dataset pred, TmpDistances distances,
                    int n_lib, int n_pred, int E, int tau)
{
    using simd_t = simd::simd<float, simd::simd_abi::native>;

    Kokkos::parallel_for(
        "EDM::knn::calc_distances", Kokkos::TeamPolicy<>(n_pred, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            const int i = member.league_rank();

#ifdef USE_SIMD_PRIMITIVES
            // Vectorized loop
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib / simd_t::size()),
                [=](int j) {
                    simd_t dist = simd_t(0.0f);

                    for (int e = 0; e < E; e++) {
                        for (size_t k = 0; k < lib.extent(1); k++) {
                            simd_t diff =
                                simd_t(pred(i + e * tau, k)) -
                                simd_t(&lib(j * simd_t::size() + e * tau, k),
                                       simd::element_aligned_tag());
                            dist += diff * diff;
                        }
                    }

                    dist.copy_to(&distances(i, j * simd_t::size()),
                                 simd::element_aligned_tag());
                });
#endif

            // Remainder loop
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member,
#ifdef USE_SIMD_PRIMITIVES
                                        n_lib / simd_t::size() * simd_t::size(),
#else
                                        0,
#endif
                                        n_lib),
                [=](int j) {
                    float dist = 0.0f;

                    for (int e = 0; e < E; e++) {
                        for (size_t k = 0; k < lib.extent(1); k++) {
                            float diff =
                                pred(i + e * tau, k) - lib(j + e * tau, k);
                            dist += diff * diff;
                        }
                    }

                    distances(i, j) = dist;
                });

            // Ignore degenerate neighbors
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n_lib),
                                 [=](int j) {
                                     if (distances(i, j) == 0.0f) {
                                         distances(i, j) = FLT_MAX;
                                     }
                                 });
        });
}

void partial_sort(TmpDistances distances, SimplexLUT out, int n_lib, int n_pred,
                  int top_k, int shift)
{
#ifdef KOKKOS_ENABLE_CUDA
    // Make sure a thread sees at least top_k elements
    const int team_size = std::min(32, std::max(n_lib / top_k, 1));
#else
    const int team_size = 1;
    using std::min;
#endif

    const size_t scratch_size =
        ScratchDistances::shmem_size(team_size, top_k) +
        ScratchIndices::shmem_size(team_size, top_k) +
        Kokkos::View<int *, DevScratchSpace,
                     Kokkos::MemoryUnmanaged>::shmem_size(team_size);

    // Partially sort each row
    Kokkos::parallel_for(
        "EDM::knn::partial_sort",
        Kokkos::TeamPolicy<>(n_pred, team_size)
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
                Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
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

void knn(TimeSeries lib, TimeSeries pred, SimplexLUT out, TmpDistances tmp,
         int E, int tau, int Tp, int top_k)
{
    Kokkos::Profiling::pushRegion("EDM::knn");

    const int n_partial = (E - 1) * tau;
    const int n_lib = lib.extent(0) - n_partial - Tp;
    const int n_pred = pred.extent(0) - n_partial;

    if (E <= 0) {
        throw std::invalid_argument("E must be greater than zero");
    } else if (tau <= 0) {
        throw std::invalid_argument("tau must be greater than zero");
    } else if (Tp < 0) {
        throw std::invalid_argument("Tp must be greater or equal to zero");
    } else if (top_k <= 0) {
        throw std::invalid_argument("top_k must be greater than zero");
    } else if (n_lib <= 0 || n_lib < top_k) {
        throw std::invalid_argument("lib size is too small");
    } else if (n_pred <= 0) {
        throw std::invalid_argument("pred size is too small");
    } else if (tmp.extent_int(0) < n_pred || tmp.extent_int(1) < n_lib) {
        throw std::invalid_argument(
            "TmpDistances must be larger or equal to (n_pred, n_lib)");
    } else if (out.distances.extent_int(0) != n_pred ||
               out.distances.extent_int(1) != top_k) {
        throw std::invalid_argument("LUT must have shape (n_pred, top_k)");
    }

    // Calculate all-to-all distances
    calc_distances(lib, pred, tmp, n_lib, n_pred, E, tau);

    // Sort the distance matrix
    partial_sort(tmp, out, n_lib, n_pred, top_k, n_partial + Tp);

    Kokkos::Profiling::popRegion();
}

void knn(Dataset lib, Dataset pred, SimplexLUT out, TmpDistances tmp, int E,
         int tau, int Tp, int top_k)
{
    Kokkos::Profiling::pushRegion("EDM::knn");

    const int n_partial = (E - 1) * tau;
    const int n_lib = lib.extent(0) - n_partial - Tp;
    const int n_pred = pred.extent(0) - n_partial;

    if (E <= 0) {
        throw std::invalid_argument("E must be greater than zero");
    } else if (tau <= 0) {
        throw std::invalid_argument("tau must be greater than zero");
    } else if (Tp < 0) {
        throw std::invalid_argument("Tp must be greater or equal to zero");
    } else if (top_k <= 0) {
        throw std::invalid_argument("top_k must be greater than zero");
    } else if (n_lib <= 0 || n_lib < top_k) {
        throw std::invalid_argument("lib size is too small");
    } else if (n_pred <= 0) {
        throw std::invalid_argument("pred size is too small");
    } else if (tmp.extent_int(0) < n_pred || tmp.extent_int(1) < n_lib) {
        throw std::invalid_argument(
            "TmpDistances must be larger or equal to (n_pred, n_lib)");
    } else if (out.distances.extent_int(0) != n_pred ||
               out.distances.extent_int(1) != top_k) {
        throw std::invalid_argument("LUT must have shape (n_pred, top_k)");
    } else if (lib.extent(1) != pred.extent(1)) {
        throw std::invalid_argument(
            "lib and pred must have same number of time series");
    }

    // Calculate all-to-all distances
    calc_distances(lib, pred, tmp, n_lib, n_pred, E, tau);

    // Sort the distance matrix
    partial_sort(tmp, out, n_lib, n_pred, top_k, n_partial + Tp);

    Kokkos::Profiling::popRegion();
}

void normalize_lut(SimplexLUT lut)
{
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
                min_dist = fmin(min_dist, dist);
                max_dist = fmax(max_dist, dist);
            }

            for (int j = 0; j < top_k; j++) {
                const float dist = distances(i, j);

                float weighted_dist = 0.0f;

                if (min_dist > 0.0f) {
                    weighted_dist = exp(-dist / min_dist);
                } else {
                    weighted_dist = dist > 0.0f ? 0.0f : 1.0f;
                }

                const float weight = fmax(weighted_dist, MIN_WEIGHT);

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
