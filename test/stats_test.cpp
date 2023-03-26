#include <doctest/doctest.h>

#include "../src/stats.hpp"
#include "../src/types.hpp"

namespace edm
{

TEST_CASE("Compute Pearson's correlation coefficient, MAE and MSE")
{
    const int N = 1000;

    MutableTimeSeries x("x", N), y("y", N);

    auto x_mirror = Kokkos::create_mirror(x);
    auto y_mirror = Kokkos::create_mirror(y);

    for (int i = 0; i < N; i++) {
        x_mirror(i) = i;
        y_mirror(i) = i;
    }

    Kokkos::deep_copy(x, x_mirror);
    Kokkos::deep_copy(y, y_mirror);

    CHECK(corrcoef(x, y) == doctest::Approx(1.0f));
    CHECK(mae(x, y) == doctest::Approx(0.0f));
    CHECK(mse(x, y) == doctest::Approx(0.0f));
}

} // namespace edm
