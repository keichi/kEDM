#define DOCTEST_CONFIG_IMPLEMENT
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
    const auto L = 100;

    const auto ds1 = load_csv("logistic_map.csv");
    const auto library = TimeSeries(ds1, Kokkos::pair<int, int>(0, L), 0);
    const auto target = TimeSeries(ds1, Kokkos::pair<int, int>(L, L * 2), 0);
    const auto prediction = MutableTimeSeries("prediction", L);

    const auto ds2 = load_csv("logistic_map_validation.csv");
    const auto ds2_mirror =
        Kokkos::create_mirror_view_and_copy(HostSpace(), ds2);

    for (size_t i = 0; i < ds2_mirror.extent(0); i++) {
        const auto theta = ds2_mirror(i, 0);

        smap(prediction, library, target, E, tau, Tp, theta);

        const auto shift = (E - 1) * tau + Tp;
        const auto shifted_target =
            TimeSeries(target, Kokkos::pair<int, int>(shift, L));

        // TODO Need to check why the test fails without this fence
        Kokkos::fence();

        CHECK(corrcoef(shifted_target, prediction) ==
              doctest::Approx(ds2_mirror(i, 1)).epsilon(0.01f));
    }
}

} // namespace edm

int main(int argc, char **argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    return doctest::Context(argc, argv).run();
}
