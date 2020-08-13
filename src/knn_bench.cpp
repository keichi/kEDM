#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../src/knn.hpp"
#include "../src/types.hpp"

const uint32_t L = 10000;
const uint32_t E = 1;
const uint32_t tau = 1;
const uint32_t Tp = 1;

int main(int argc, char *argv[])
{
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

    for (auto i = 0; i < 10; i++) {
        edm::knn(library, target, lut_out, lut_tmp, E, tau, Tp, E + 1);
    }

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;

    return 0;
}
