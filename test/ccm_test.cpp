#include <doctest/doctest.h>

#include "../src/ccm.hpp"
#include "../src/io.hpp"
#include "../src/types.hpp"

namespace edm
{

TEST_CASE("Compute Convergent Cross Mapping")
{
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

    const auto rhos1 = ccm(anchovy, sst, lib_sizes, sample, E, tau, Tp, 42);
    const auto rhos2 = ccm(sst, anchovy, lib_sizes, sample, E, tau, Tp, 42);

    const Dataset ds2 = load_csv("anchovy_sst_ccm_validation.csv");
    const auto valid_rhos1 = Kokkos::create_mirror_view_and_copy(
        HostSpace(), Kokkos::subview(ds2, Kokkos::ALL, 1));
    const auto valid_rhos2 = Kokkos::create_mirror_view_and_copy(
        HostSpace(), Kokkos::subview(ds2, Kokkos::ALL, 2));

    for (size_t i = 0; i < rhos1.size(); i++) {
        CHECK(rhos1[i] == doctest::Approx(valid_rhos1(i)));
        CHECK(rhos2[i] == doctest::Approx(valid_rhos2(i)));
    }
}

} // namespace edm
