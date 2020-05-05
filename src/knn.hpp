#ifndef __KNN_HPP__
#define __KNN_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

const auto MIN_WEIGHT = 1e-6f;

class NearestNeighbors
{
    LUT _cache;

public:
    NearestNeighbors(LUT cache) : _cache(cache) {}

    void run(const TimeSeries library, const TimeSeries target, LUT &lut,
             uint32_t E, uint32_t tau, uint32_t Tp, uint32_t top_k)
    {
        using std::max;
        using std::min;

        const auto shift = (E - 1) * tau + Tp;

        const uint32_t n_library = library.size() - shift;
        const uint32_t n_target = target.size() - shift + Tp;

        auto distances = _cache.distances;
        auto indices = _cache.indices;

        // Compute all-to-all distances
        // MDRange parallel version
        Kokkos::parallel_for(
            "calc_distances",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                   {n_target, n_library}),
            KOKKOS_LAMBDA(int i, int j) {
                distances(i, j) = 0.0f;

                for (auto e = 0u; e < E; e++) {
                    const auto diff =
                        target(i + e * tau) - library(j + e * tau);
                    distances(i, j) += diff * diff;
                }

                indices(i, j) = j;
            });

        // Ignore degenerate neighbors
        Kokkos::parallel_for(
            "ignore_degenerates",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                                   {n_target, n_library}),
            KOKKOS_LAMBDA(int i, int j) {
                if (target.data() + i == library.data() + j) {
                    distances(i, j) = FLT_MAX;
                }
            });

        // Partially sort each row
        Kokkos::parallel_for(
            "partial_sort", n_target, KOKKOS_LAMBDA(const int i) {
                for (auto j = 1u; j < n_library; j++) {
                    auto cur_dist = distances(i, j);
                    auto cur_idx = indices(i, j);

                    // Skip elements larger than the current k-th smallest
                    // element
                    if (j >= top_k && cur_dist > distances(i, top_k - 1)) {
                        continue;
                    }

                    auto k = 0u;
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
                          Kokkos::subview(distances,
                                          std::make_pair(0u, n_target),
                                          std::make_pair(0u, top_k)));
        Kokkos::deep_copy(lut.indices,
                          Kokkos::subview(indices, std::make_pair(0u, n_target),
                                          std::make_pair(0u, top_k)));
    }
};

void normalize_lut(LUT &lut)
{
    using std::max;
    using std::min;
    using std::sqrt;

    auto distances = lut.distances;
    auto indices = lut.indices;
    const auto L = distances.extent(0);
    const auto top_k = distances.extent(1);

    // Normalize lookup table
    Kokkos::parallel_for(
        "normalize_distances", L, KOKKOS_LAMBDA(int i) {
            auto sum_weights = 0.0f;
            auto min_dist = FLT_MAX;
            auto max_dist = 0.0f;

            for (auto j = 0u; j < top_k; j++) {
                const auto dist = distances(i, j);

                min_dist = min(min_dist, dist);
                max_dist = max(max_dist, dist);

                distances(i, j) = dist;
            }

            for (auto j = 0u; j < top_k; j++) {
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

            for (auto j = 0u; j < top_k; j++) {
                distances(i, j) /= sum_weights;
            }
        });
}

} // namespace edm

#endif
