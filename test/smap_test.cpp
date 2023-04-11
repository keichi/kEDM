#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/smap.hpp"
#include "../src/stats.hpp"
#include "../src/types.hpp"

namespace edm
{

TEST_CASE("Compute S-Map projections")
{
    const auto E = 2, tau = 1, Tp = 1;
    std::vector<float> thetas = {0.0, 0.01, 0.1, 0.3, 0.5, 0.75, 1, 2,
                                 3,   4,    5,   6,   7,   8,    9};

    const auto ds1 = load_csv("logistic_map.csv");
    const auto lib = TimeSeries(ds1, Kokkos::ALL, 1);
    const auto pred = TimeSeries(ds1, Kokkos::ALL, 1);
    const auto result =
        MutableTimeSeries("result", pred.extent(0) - (E - 1) * tau);

    const auto ds2 = load_csv("smap_valid_logistic_map.csv");
    const auto ds2_mirror =
        Kokkos::create_mirror_view_and_copy(HostSpace(), ds2);

    for (size_t i = 0; i < thetas.size(); i++) {
        smap(result, lib, pred, pred, E, tau, Tp, thetas[i]);

        const auto result_mirror =
            Kokkos::create_mirror_view_and_copy(HostSpace(), result);

        for (size_t j = 0; j < result_mirror.extent(0); j++) {
            CHECK(result_mirror(j) ==
                  doctest::Approx(ds2(j, i + 1)).epsilon(5e-2f));
        }
    }
}

} // namespace edm
