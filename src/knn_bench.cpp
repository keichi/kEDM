#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "../src/knn.hpp"
#include "../src/types.hpp"

int main(int argc, char *argv[])
{
    cxxopts::Options options("knn-bench", "Benchmark k-nearest neighbor kernel");
    options.add_options()
      ("e,embedding-dim", "Embedding dimension", cxxopts::value<int>()->default_value("20"))
      ("l,length", "Length of time series", cxxopts::value<int>()->default_value("10000"))
      ("i,iteration", "Number of iterations", cxxopts::value<int>()->default_value("10"))
      ("t,tau", "Time delay", cxxopts::value<int>()->default_value("1"))
      ("h,help", "Print usage");

    const auto result = options.parse(argc, argv);

    const auto L = result["length"].as<int>();
    const auto E = result["embedding-dim"].as<int>();
    const auto iterations = result["iteration"].as<int>();
    const auto tau = result["tau"].as<int>();
    const auto Tp = 1;

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
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

    edm::LUT lut_tmp(L, L);
    edm::LUT lut_out(L - (E - 1) * tau, E + 1);

    Kokkos::Timer timer;

    for (auto i = 0; i < iterations; i++) {
        edm::knn(library, target, lut_out, lut_tmp, E, tau, Tp, E + 1);
    }

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;

    return 0;
}
