#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <argh.h>

#include "smap.hpp"
#include "stats.hpp"
#include "timer.hpp"
#include "types.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": S-Map Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -l, --length arg        Length of time series (default: 10,000)\n"
        "  -e, --embedding-dim arg Embedding dimension (default: 20)\n"
        "  -t, --tau arg           Time delay (default: 1)\n"
        "  --theta arg             Localization parameter (default: 1.0)\n"
        "  -i, --iteration arg     Number of iterations (default: 10)\n"
        "  -v, --verbose           Enable verbose logging (default: false)\n"
        "  -h, --help              Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-l", "--length", "-e", "--embedding-dim", "-t", "--tau",
                       "--theta", "-i", "--iteration"});
    cmdl.parse(argc, argv);

    int L;
    cmdl({"l", "length"}, 10000) >> L;
    int E;
    cmdl({"e", "embedding-dim"}, 20) >> E;
    int tau;
    cmdl({"t", "tau"}, 1) >> tau;
    float theta;
    cmdl({"theta"}, 1.0) >> theta;
    int iterations;
    cmdl({"i", "iteration"}, 10) >> iterations;
    int Tp = 1;

    if (cmdl[{"-h", "--help"}]) {
        usage(cmdl[0]);
        return 0;
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    edm::MutableTimeSeries lib("lib", L);
    edm::MutableTimeSeries pred("pred", L);
    edm::MutableTimeSeries result("result", L);

    auto lib_mirror = Kokkos::create_mirror_view(lib);
    auto pred_mirror = Kokkos::create_mirror_view(pred);

    lib_mirror(0) = 0.23456f;
    pred_mirror(0) = 0.34567f;

    for (int i = 1; i < L; i++) {
        lib_mirror(i) = 4.0f * lib_mirror(i - 1) * (1.0f - lib_mirror(i - 1));
        pred_mirror(i) =
            4.0f * pred_mirror(i - 1) * (1.0f - pred_mirror(i - 1));
    }

    Kokkos::deep_copy(lib, lib_mirror);
    Kokkos::deep_copy(pred, pred_mirror);

    Kokkos::fence();

    Kokkos::Timer timer;

    for (auto i = 0; i < iterations; i++) {
        edm::smap(result, lib, pred, lib, E, tau, Tp, theta);
    }

    Kokkos::fence();

    std::cout << timer.seconds() / iterations << " [s]" << std::endl;

    return 0;
}
