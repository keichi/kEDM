#include <iostream>

// #include <Kokkos_Random.hpp>

#include "smap.hpp"
#include "stats.hpp"
#include "timer.hpp"
#include "types.hpp"

int main(int argc, char *argv[])
{
    const int L = 10000;

    Kokkos::ScopeGuard kokkos(argc, argv);

    edm::MutableTimeSeries library("library", L);
    edm::MutableTimeSeries target("target", L);
    edm::MutableTimeSeries prediction("prediction", L);

    auto library_mirror = Kokkos::create_mirror_view(library);
    auto target_mirror = Kokkos::create_mirror_view(target);

    // Kokkos::Random_XorShift64_Pool<> rand_pool(1931);
    // Kokkos::fill_random(library, rand_pool, 1.0);
    // Kokkos::fill_random(target, rand_pool, 1.0);

    library_mirror(0) = 0.23456f;
    target_mirror(0) = 0.34567f;

    for (int i = 1; i < L; i++) {
        library_mirror(i) =
            4.0f * library_mirror(i - 1) * (1.0f - library_mirror(i - 1));
        target_mirror(i) =
            4.0f * target_mirror(i - 1) * (1.0f - target_mirror(i - 1));
    }

    Kokkos::deep_copy(library, library_mirror);
    Kokkos::deep_copy(target, target_mirror);

    std::cout << "theta\trho\truntime" << std::endl;

    for (float theta = 0.0f; theta < 3.0f; theta += 0.1f) {
        Kokkos::Timer timer;

        edm::smap(prediction, library, target, 2, 1, 1, theta);

        auto shift = (2 - 1) * 1 + 1;
        auto shifted_target =
            Kokkos::subview(target, Kokkos::pair<int, int>(shift, L - shift));

        std::cout << theta << "\t" << edm::corrcoef(shifted_target, prediction)
                  << "\t" << timer.seconds() << std::endl;
    }

    return 0;
}
