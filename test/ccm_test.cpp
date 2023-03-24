#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/ccm.hpp"
#include "../src/types.hpp"

namespace edm
{

TEST_CASE("Compute Convergent Cross Mapping") {
    const int E = 3;
    const int tau = 1;
    const int Tp = 0;
    const int sample = 100;

    std::vector<int> lib_sizes;
    for (int i = 10; i <= 75; i += 5) {
        lib_sizes.push_back(i);
    }

    const Dataset ds1 = load_csv("sardine_anchovy_sst.csv");
    const auto anchovy = Kokkos::subview(ds1, Kokkos::ALL, 1);
    const auto sst = Kokkos::subview(ds1, Kokkos::ALL, 4);

    ccm(anchovy, sst, lib_sizes, sample, E, tau, Tp);
    ccm(sst, anchovy, lib_sizes, sample, E, tau, Tp);
}

} // namespace edm
