#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <argh.h>

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

#include "ccm.hpp"
#include "timer.hpp"
#include "types.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": Partial Sort Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -n, --size arg          Matrix size (default: 10,000)\n"
        "  -k, --topk arg          Number of top-k neighbors (default: 21)\n"
        "  -i, --iteration arg     Number of iterations (default: 10)\n"
        "  -f, --full-sort         Use full sort instead of partial sort\n"
        "  -v, --verbose           Enable verbose logging (default: false)\n"
        "  -h, --help              Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-n", "--size", "-k", "--topk", "-i", "--iteration"});
    cmdl.parse(argc, argv);

    int N;
    cmdl({"n", "size"}, 10000) >> N;
    int k;
    cmdl({"k", "topk"}, 21) >> k;
    int iterations;
    cmdl({"i", "iteration"}, 10) >> iterations;
    bool full = cmdl[{"-f", "--full-sort"}];
    bool verbose = cmdl[{"-v", "--verbose"}];

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    if (verbose) {
        std::cout << "N: " << N << std::endl;
        std::cout << "k: " << k << std::endl;
        std::cout << "iterations: " << iterations << std::endl;
        std::cout << "full_sort: " << (full ? "true" : "false") << std::endl;
    }

    edm::TmpDistances distances("distances", N, N);
    edm::TmpIndices indices("indices", N, N);

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);

    Kokkos::Timer timer;
    Timer timer_sort;

    LIKWID_MARKER_INIT;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_REGISTER("partial_sort");
        LIKWID_MARKER_REGISTER("full_sort");
    }

    for (auto i = 0; i < iterations; i++) {
        // Fill distances with random values for each iteration
        Kokkos::fill_random(distances, rand_pool, 1.0f);
        Kokkos::fence();

        timer_sort.start();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            if (full) {
                LIKWID_MARKER_START("full_sort");
            } else {
                LIKWID_MARKER_START("partial_sort");
            }
        }

        if (full) {
            edm::full_sort(distances, indices, N, N, 0, 0);
        } else {
            edm::partial_sort(distances, indices, k, N, N, 0, 0);
        }

        Kokkos::fence();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            if (full) {
                LIKWID_MARKER_STOP("full_sort");
            } else {
                LIKWID_MARKER_STOP("partial_sort");
            }
        }

        timer_sort.stop();
    }

    LIKWID_MARKER_CLOSE;

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;

    if (full) {
        std::cout << "full_sort " << timer_sort.elapsed() / iterations
                  << std::endl;
    } else {
        std::cout << "partial_sort " << timer_sort.elapsed() / iterations
                  << std::endl;
    }

    return 0;
}
