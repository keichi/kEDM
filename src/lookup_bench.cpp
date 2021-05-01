#include <iostream>
#include <random>
#include <string>
#include <vector>

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

#include "knn.hpp"
#include "timer.hpp"
#include "types.hpp"
#include "xmap.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": kNN Lookup Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -l, --length arg        Length of time series (default: 10,000)\n"
        "  -n, --num-ts arg        Number of time series (default: 10,000)\n"
        "  -e, --embedding-dim arg Embedding dimension (default: 20)\n"
        "  -t, --tau arg           Time delay (default: 1)\n"
        "  -i, --iteration arg     Number of iterations (default: 10)\n"
        "  -v, --verbose           Enable verbose logging (default: false)\n"
        "  -h, --help              Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-l", "--length", "-n", "--num-ts", "-e",
                       "--embedding-dim", "-t", "--tau", "-i", "--iteration"});
    cmdl.parse(argc, argv);

    int L;
    cmdl({"l", "length"}, 10000) >> L;
    int N;
    cmdl({"n", "num-ts"}, 10000) >> N;
    int E;
    cmdl({"e", "embedding-dim"}, 20) >> E;
    int tau;
    cmdl({"t", "tau"}, 1) >> tau;
    int iterations;
    cmdl({"i", "iteration"}, 10) >> iterations;
    int Tp = 1;

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    edm::MutableDataset dataset("dataset", L, N);

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    Kokkos::fill_random(dataset, rand_pool, 1.0);

    edm::TimeSeries library(dataset, Kokkos::ALL, 0);
    edm::TimeSeries target(dataset, Kokkos::ALL, 0);

    edm::TmpDistances tmp("tmp_distances", L, L);
    edm::LUT lut(L - (E - 1) * tau, E + 1);

    edm::CrossMap ccm("ccm", N);
    edm::Targets targets("targets", N);

    Kokkos::parallel_for(
        N, KOKKOS_LAMBDA(int i) { targets(i) = i; });

    Kokkos::Timer timer;
    Timer timer_lookup;

    LIKWID_MARKER_INIT;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        LIKWID_MARKER_THREADINIT;

        LIKWID_MARKER_REGISTER("lookup");
    }

    for (auto i = 0; i < iterations; i++) {
        edm::knn(library, target, lut, tmp, E, tau, Tp, E + 1);

        Kokkos::fence();

        timer_lookup.start();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            LIKWID_MARKER_START("lookup");
        }

        edm::_xmap(ccm, dataset, lut, targets, E, tau, Tp);

        Kokkos::fence();

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            LIKWID_MARKER_STOP("lookup");
        }

        timer_lookup.stop();
    }

    LIKWID_MARKER_CLOSE;

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;
    std::cout << "lookup " << timer_lookup.elapsed() / iterations << std::endl;

    return 0;
}
