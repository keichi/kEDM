#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/types.hpp"

void test_knn_common(const int E, const int tau, const int Tp, const int top_k)
{
    edm::Dataset ds = edm::load_csv("knn_test_data.csv");
    edm::TimeSeries ts = edm::TimeSeries(ds, Kokkos::ALL, 0);

    edm::LUT cache(ts.size(), ts.size());
    edm::NearestNeighbors knn(cache);

    edm::LUT out(ts.size() - (E - 1) * tau, top_k);
    knn.run(ts, ts, out, E, tau, Tp, top_k);

    edm::Dataset validation =
        edm::load_csv("knn_test_validation_E" + std::to_string(E) + ".csv");

    CHECK(out.distances.extent(0) == validation.extent(0));
    CHECK(out.distances.extent(1) == validation.extent(1));

    CHECK(out.indices.extent(0) == validation.extent(0));
    CHECK(out.indices.extent(1) == validation.extent(1));

    for (auto row = 0u; row < out.distances.extent(0); row++) {
        for (auto col = 0u; col < out.distances.extent(1); col++) {
            CHECK(out.distances(row, col) ==
                  doctest::Approx(validation(row, col)));
        }
    }
}

TEST_CASE("Compute kNN table for E=2")
{
    Kokkos::initialize();

    test_knn_common(2, 1, 0, 4);

    Kokkos::finalize();
}

TEST_CASE("Compute kNN table for E=3")
{
    Kokkos::initialize();

    test_knn_common(3, 1, 0, 4);

    Kokkos::finalize();
}

TEST_CASE("Compute kNN table for E=4")
{
    Kokkos::initialize();

    test_knn_common(4, 1, 0, 4);

    Kokkos::finalize();
}

TEST_CASE("Compute kNN table for E=5")
{
    Kokkos::initialize();

    test_knn_common(5, 1, 0, 4);

    Kokkos::finalize();
}
