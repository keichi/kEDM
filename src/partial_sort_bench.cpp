#include <iomanip>
#include <iostream>
#include <vector>

#include <Kokkos_Random.hpp>
#include <argh.h>

#include "ccm.hpp"
#include "timer.hpp"
#include "types.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Partial Sort Benchmark (Radix Sort vs Bitonic Top-K)\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -i, --iteration arg  Number of iterations (default: 10)\n"
        "  -v, --verbose        Enable verbose logging (default: false)\n"
        "  -h, --help           Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-i", "--iteration"});
    cmdl.parse(argc, argv);

    int iterations;
    cmdl({"i", "iteration"}, 10) >> iterations;
    bool verbose = cmdl[{"-v", "--verbose"}];

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    // Test parameters
    std::vector<int> k_values = {8, 16, 32, 64, 128};
    std::vector<int> L_values = {1000, 10000, 50000};

    // Print CSV header
    std::cout << "k,L,n_pred,radix_ms,bitonic_ms,speedup" << std::endl;

    Kokkos::Random_XorShift64_Pool<> rand_pool(42);

    for (int L : L_values) {
        for (int k : k_values) {
            if (k >= L) {
                continue;
            }

            int n_lib = L;
            int n_pred = L;
            int n_partial = 0;
            int Tp = 1;

            // Allocate LUTs
            edm::SimplexLUT lut_radix(n_pred, n_lib);
            edm::SimplexLUT lut_bitonic(n_pred, n_lib);

            // Generate random distances
            Kokkos::fill_random(lut_radix.distances, rand_pool, 1e6f);

            Timer timer_radix;
            Timer timer_bitonic;

            for (int iter = 0; iter < iterations; iter++) {
                // Copy fresh data for radix sort
                Kokkos::deep_copy(lut_radix.distances, lut_radix.distances);

                timer_radix.start();
                edm::partial_sort_radix(lut_radix, k, n_lib, n_pred, n_partial,
                                        Tp);
                Kokkos::fence();
                timer_radix.stop();

                // Reset data for bitonic sort
                Kokkos::fill_random(lut_radix.distances, rand_pool, 1e6f);
                Kokkos::deep_copy(lut_bitonic.distances, lut_radix.distances);

                timer_bitonic.start();
                edm::partial_sort_bitonic(lut_bitonic, k, n_lib, n_pred,
                                          n_partial, Tp);
                Kokkos::fence();
                timer_bitonic.stop();

                // Verify results match (only on first iteration)
                if (iter == 0 && verbose) {
                    auto h_radix_dist =
                        Kokkos::create_mirror_view(lut_radix.distances);
                    auto h_bitonic_dist =
                        Kokkos::create_mirror_view(lut_bitonic.distances);
                    Kokkos::deep_copy(h_radix_dist, lut_radix.distances);
                    Kokkos::deep_copy(h_bitonic_dist, lut_bitonic.distances);

                    bool match = true;
                    for (int i = 0; i < n_pred && match; i++) {
                        for (int j = 0; j < k && match; j++) {
                            float diff = std::abs(h_radix_dist(i, j) -
                                                  h_bitonic_dist(i, j));
                            if (diff > 1e-5f) {
                                std::cerr << "Mismatch at (" << i << "," << j
                                          << "): radix=" << h_radix_dist(i, j)
                                          << " bitonic=" << h_bitonic_dist(i, j)
                                          << std::endl;
                                match = false;
                            }
                        }
                    }
                    if (!match) {
                        std::cerr << "WARNING: Results do not match for k=" << k
                                  << ", L=" << L << std::endl;
                    }
                }
            }

            double radix_ms = timer_radix.elapsed() / iterations;
            double bitonic_ms = timer_bitonic.elapsed() / iterations;
            double speedup = radix_ms / bitonic_ms;

            std::cout << std::fixed << std::setprecision(4) << k << "," << L
                      << "," << n_pred << "," << radix_ms << "," << bitonic_ms
                      << "," << speedup << std::endl;
        }
    }

    return 0;
}
