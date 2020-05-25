#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/types.hpp"

namespace edm
{

void test_knn_common(int E)
{
    const auto tau = 1;
    const auto Tp = 0;
    const auto top_k = 4;

    Dataset ds = load_csv("knn_test_data.csv");
    TimeSeries ts(ds, Kokkos::ALL, 0);

    LUT cache(ts.size(), ts.size());
    NearestNeighbors knn(cache);

    LUT lut(ts.size() - (E - 1) * tau, top_k);
    knn.run(ts, ts, lut, E, tau, Tp, top_k);

    Dataset d_validation =
        load_csv("knn_test_validation_E" + std::to_string(E) + ".csv");

    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    auto indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.indices);
    auto validation =
        Kokkos::create_mirror_view_and_copy(HostSpace(), d_validation);

    CHECK(distances.extent(0) == validation.extent(0));
    CHECK(distances.extent(1) == validation.extent(1));

    CHECK(indices.extent(0) == validation.extent(0));
    CHECK(indices.extent(1) == validation.extent(1));

    for (auto row = 0u; row < distances.extent(0); row++) {
        for (auto col = 0u; col < distances.extent(1); col++) {
            CHECK(distances(row, col) == doctest::Approx(validation(row, col)));
        }
    }

    normalize_lut(lut);

    Kokkos::deep_copy(distances, lut.distances);

    for (auto row = 0u; row < distances.extent(0); row++) {
        auto sum = 0.0f;
        for (auto col = 0u; col < distances.extent(1); col++) {
            sum += distances(row, col);
        }

        CHECK(sum == doctest::Approx(1.0f));
    }
}

TEST_CASE("Compute kNN table for E=2") { test_knn_common(2); }

TEST_CASE("Compute kNN table for E=3") { test_knn_common(3); }

TEST_CASE("Compute kNN table for E=4") { test_knn_common(4); }

TEST_CASE("Compute kNN table for E=5") { test_knn_common(5); }

} // namespace edm

int main(int argc, char **argv)
{
    Kokkos::initialize();

    int res = doctest::Context(argc, argv).run();

    Kokkos::finalize();

    return res;
}
