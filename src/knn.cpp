#include <cassert>

#include <Kokkos_Core.hpp>

#include "knn.hpp"
#include "types.hpp"

namespace edm
{

void NearestNeighbors::run(const TimeSeries &library, const TimeSeries &target,
                           LUT &lut, int E, int tau, int Tp, int top_k)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::max;
    using std::min;
#endif

    assert(E > 0 && tau > 0 && Tp >= 0 && top_k > 0);

    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    assert(n_library > 0 && n_target > 0);

    auto distances = _cache.distances;
    auto indices = _cache.indices;

    assert(distances.extent(0) >= n_target && distances.extent(1) >= n_library);

    // Compute all-to-all distances
    // MDRange parallel version
    // Kokkos::parallel_for(
    //     "calc_distances",
    //     Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_target,
    //     n_library}), KOKKOS_LAMBDA(int i, int j) {
    //         distances(i, j) = 0.0f;
    //
    //         for (auto e = 0; e < E; e++) {
    //             auto diff = target(i + e * tau) - library(j + e * tau);
    //             distances(i, j) += diff * diff;
    //         }
    //
    //         indices(i, j) = j;
    //     });

    Kokkos::parallel_for(
        "calc_distances", Kokkos::TeamPolicy<>(n_library, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int j = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_target), [=](size_t i) {
                    float dist = 0.0f;

                    for (auto e = 0; e < E; e++) {
                        float diff = target(i + e * tau) - library(j + e * tau);
                        dist += diff * diff;
                    }

                    distances(i, j) = dist;
                    indices(i, j) = j;
                });
        });

    // Ignore degenerate neighbors
    // Kokkos::parallel_for(
    //     "ignore_degenerates",
    //     Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_target,
    //     n_library}), KOKKOS_LAMBDA(int i, int j) {
    //         if (target.data() + i == library.data() + j) {
    //             distances(i, j) = FLT_MAX;
    //         }
    //     });

    Kokkos::parallel_for(
        "ignore_degenerates", Kokkos::TeamPolicy<>(n_target, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int j = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_library), [=](size_t i) {
                    if (library.data() + i == target.data() + j) {
                        distances(i, j) = FLT_MAX;
                    }
                });
        });

    // Partially sort each row
    Kokkos::parallel_for(
        "partial_sort", n_target, KOKKOS_LAMBDA(int i) {
            for (auto j = 1; j < n_library; j++) {
                auto cur_dist = distances(i, j);
                auto cur_idx = indices(i, j);

                // Skip elements larger than the current k-th smallest
                // element
                if (j >= top_k && cur_dist > distances(i, top_k - 1)) {
                    continue;
                }

                auto k = 0;
                // Shift elements until the insertion point is found
                for (k = min(j, top_k - 1); k > 0; k--) {
                    if (distances(i, k - 1) <= cur_dist) {
                        break;
                    }

                    // Shift element
                    distances(i, k) = distances(i, k - 1);
                    indices(i, k) = indices(i, k - 1);
                }

                // Insert the new element
                distances(i, k) = cur_dist;
                indices(i, k) = cur_idx;
            }
        });

    // Compute L2 norms from SSDs and shift indices
    Kokkos::parallel_for(
        "calc_norms",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n_target, top_k}),
        KOKKOS_LAMBDA(int i, int j) {
            distances(i, j) = sqrt(distances(i, j));
            indices(i, j) += shift;
        });

    // Copy LUT from cache to output
    Kokkos::deep_copy(lut.distances,
                      Kokkos::subview(distances, std::make_pair(0, n_target),
                                      std::make_pair(0, top_k)));
    Kokkos::deep_copy(lut.indices,
                      Kokkos::subview(indices, std::make_pair(0, n_target),
                                      std::make_pair(0, top_k)));
}

void normalize_lut(LUT &lut)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::exp;
    using std::max;
    using std::min;
    using std::sqrt;
#endif

    auto distances = lut.distances;
    auto indices = lut.indices;
    const int L = distances.extent(0);
    const int top_k = distances.extent(1);

    // Normalize lookup table
    Kokkos::parallel_for(
        "normalize_distances", L, KOKKOS_LAMBDA(int i) {
            auto sum_weights = 0.0f;
            auto min_dist = FLT_MAX;
            auto max_dist = 0.0f;

            for (auto j = 0; j < top_k; j++) {
                const auto dist = distances(i, j);

                min_dist = min(min_dist, dist);
                max_dist = max(max_dist, dist);
            }

            for (auto j = 0; j < top_k; j++) {
                const auto dist = distances(i, j);

                auto weighted_dist = 0.0f;

                if (min_dist > 0.0f) {
                    weighted_dist = exp(-dist / min_dist);
                } else {
                    weighted_dist = dist > 0.0f ? 0.0f : 1.0f;
                }

                const auto weight = max(weighted_dist, MIN_WEIGHT);

                distances(i, j) = weight;

                sum_weights += weight;
            }

            for (auto j = 0; j < top_k; j++) {
                distances(i, j) /= sum_weights;
            }
        });
}

} // namespace edm
