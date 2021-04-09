#include <iostream>

#include <Kokkos_Random.hpp>

#include "smap.hpp"
#include "timer.hpp"

int main(int argc, char *argv[])
{
    const int L = 10000;

    Kokkos::ScopeGuard kokkos(argc, argv);

    edm::MutableTimeSeries library("library", L);
    edm::MutableTimeSeries target("target", L);
    edm::MutableTimeSeries prediction("prediction", L);

    Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    Kokkos::fill_random(library, rand_pool, 1.0);
    Kokkos::fill_random(target, rand_pool, 1.0);

    Kokkos::Timer timer;

    edm::smap(prediction, library, target, 20, 1, 1, 1.0f);

    std::cout << "elapsed: " << timer.seconds() << " [s]" << std::endl;

    return 0;
}
