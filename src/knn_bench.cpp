#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <argh.h>

#include "../src/knn.hpp"
#include "../src/types.hpp"

void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": k-Nearest Neighbors Search Benchmark\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...]\n"
        "  -l, --length arg        Length of time series (default: 10,000)\n"
        "  -e, --embedding-dim arg Embedding dimension (default: 20)\n"
        "  -t, --tau arg           Time delay (default: 1)\n"
        "  -i, --iteration arg     Number of iterations (default: 10)\n"
        "  -x, --kernel arg        Kernel type {cpu|gpu} (default: cpu)\n"
        "  -v, --verbose           Enable verbose logging (default: false)\n"
        "  -h, --help              Show this help";

    std::cout << msg << std::endl;
}

int main(int argc, char *argv[])
{
    argh::parser cmdl({"-l", "--length", "-e", "--embedding-dim", "-t", "--tau",
                       "-i", "--iteration"});
    cmdl.parse(argc, argv);

    int L;
    cmdl({"l", "length"}, 10000) >> L;
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

    edm::MutableTimeSeries library("library", L);
    edm::MutableTimeSeries target("target", L);

    auto h_library = Kokkos::create_mirror_view(library);
    auto h_target = Kokkos::create_mirror_view(target);

    std::random_device rand_dev;
    std::default_random_engine engine(rand_dev());
    std::uniform_real_distribution<> dist(0.0f, 1.0f);

    for (auto i = 0; i < L; i++) {
        h_library(i) = dist(engine);
        h_target(i) = dist(engine);
    }

    Kokkos::deep_copy(library, h_library);
    Kokkos::deep_copy(target, h_target);

    edm::TmpDistances tmp("tmp_distances", L, L);
    edm::LUT lut_out(L - (E - 1) * tau, E + 1);

    Kokkos::Timer timer;

    for (auto i = 0; i < iterations; i++) {
        edm::knn(library, target, lut_out, tmp, E, tau, Tp, E + 1);
    }

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;

    return 0;
}
