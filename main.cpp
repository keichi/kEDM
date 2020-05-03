#include <iostream>
#include <random>

#include <Kokkos_Core.hpp>

using std::max;
using std::min;
using std::sqrt;

void compute_knn_table(Kokkos::View<float *> ts,
                       Kokkos::View<float **> distances,
                       Kokkos::View<unsigned int **> indices, const int E,
                       const int tau)
{
    const int L = ts.extent(0);
    const int top_k = E + 1;

    // Compute all-to-all distances
    // Flat parallel version
    // Kokkos::parallel_for(
    //     "calc_distances", L - (E - 1) * tau, KOKKOS_LAMBDA(const int i) {
    //         for (int j = 0; j < L - (E - 1) * tau; j++) {
    //             for (int e = 0; e < E; e++) {
    //                 auto diff = ts(i + e * tau) - ts(j + e * tau);
    //                 distances(i, j) = diff * diff;
    //                 indices(i, j) = j;
    //             }
    //         }
    //     });

    // Compute all-to-all distances
    // Nested parallel version
    // Kokkos::parallel_for(
    //     "calc_distances", Kokkos::TeamPolicy<>(L - (E - 1) * tau, Kokkos::AUTO),
    //     KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
    //         int i = member.league_rank();

    //         Kokkos::parallel_for(
    //             Kokkos::TeamThreadRange(member, L - (E - 1) * tau), [=](int j) {
    //                 for (int e = 0; e < E; e++) {
    //                     float diff = ts(i + e * tau) - ts(j + e * tau);
    //                     distances(i, j) += diff * diff;
    //                 }
    //                 indices(i, j) = j;
    //             });
    //     });

    // Compute all-to-all distances
    // MDRange parallel version
    Kokkos::parallel_for(
        "calc_distances",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {L - (E - 1) * tau, L - (E - 1) * tau}),
        KOKKOS_LAMBDA(int i, int j) {
            for (int e = 0; e < E; e++) {
                auto diff = ts(i + e * tau) - ts(j + e * tau);
                distances(i, j) = diff * diff;
                indices(i, j) = j;
            }
        });

    // Partial sort each row
    Kokkos::parallel_for(
        "sort", L - (E - 1) * tau, KOKKOS_LAMBDA(const int i) {
            for (int j = 1; j < L - (E - 1) * tau; j++) {
                float cur_dist = distances(i, j);
                unsigned int cur_idx = indices(i, j);

                // Skip elements larger than the current k-th smallest element
                if (j >= top_k && cur_dist > distances(i, top_k - 1)) {
                    continue;
                }

                int k;
                // Shift elements until the insertion point is found
                for (k = min(j - 1, top_k - 1); k > 0; k--) {
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

    // TODO Ignore degenerate neighbors

    // Normalize lookup table
    Kokkos::parallel_for(
        "normalize", L - (E - 1) * tau, KOKKOS_LAMBDA(const int i) {
            const auto MIN_WEIGHT = 1e-6f;
            float sum_weights = 0.0f;
            float min_dist = FLT_MAX;
            float max_dist = 0.0f;

            for (int j = 0; j < top_k; j++) {
                float dist = sqrt(distances(i, j));

                min_dist = min(min_dist, dist);
                max_dist = max(max_dist, dist);

                distances(i, j) = dist;
            }

            for (int j = 0; j < top_k; j++) {
                float dist = distances(i, j);

                float weighted_dist = 0.0f;

                if (min_dist > 0.0f) {
                    weighted_dist = exp(-dist / min_dist);
                } else {
                    weighted_dist = dist > 0.0f ? 0.0f : 1.0f;
                }

                float weight = max(weighted_dist, MIN_WEIGHT);

                distances(i, j) = weight;

                sum_weights += weight;
            }

            for (int j = 0; j < top_k; j++) {
                distances(i, j) /= sum_weights;
            }
        });
}

void compute_knn_tables(Kokkos::View<float *> ts, const int E_max,
                        const int tau)
{
    const int L = ts.extent(0);

    Kokkos::View<float **> distances_tmp("distances_tmp", L, L);
    Kokkos::View<unsigned int **> indices_tmp("indices_tmp", L, L);

    std::vector<Kokkos::View<float **>> distances_all;
    std::vector<Kokkos::View<unsigned int **>> indices_all;

    for (int E = 1; E <= E_max; E++) {
        Kokkos::Timer timer;

        const int top_k = E + 1;

        compute_knn_table(ts, distances_tmp, indices_tmp, E, tau);

        Kokkos::View<float **> distances("distances", L - (E - 1) * tau, top_k);
        Kokkos::View<unsigned int **> indices("indices", L - (E - 1) * tau,
                                              top_k);

        Kokkos::deep_copy(distances,
                          Kokkos::subview(distances_tmp,
                                          std::make_pair(0, L - (E - 1) * tau),
                                          std::make_pair(0, top_k)));
        Kokkos::deep_copy(indices,
                          Kokkos::subview(indices_tmp,
                                          std::make_pair(0, L - (E - 1) * tau),
                                          std::make_pair(0, top_k)));

        distances_all.push_back(distances);
        indices_all.push_back(indices);

        std::cout << "E = " << E << " elapsed: " << timer.seconds() << " s"
                  << std::endl;
    }
}

void run()
{
    const int L = 10000;

    Kokkos::View<float *> ts("ts", L);
    Kokkos::View<float *>::HostMirror h_ts = Kokkos::create_mirror_view(ts);

    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<float> unif(-1.0f, 1.0f);

    for (size_t i = 0; i < ts.extent(0); i++) {
        h_ts(i) = unif(engine);
    }

    Kokkos::deep_copy(ts, h_ts);

    Kokkos::Timer timer;

    compute_knn_tables(ts, 20, 1);

    Kokkos::fence();

    std::cout << "elapsed: " << timer.seconds() << " s" << std::endl;
}

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);

    run();

    Kokkos::finalize();

    return 0;
}
