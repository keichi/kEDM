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

    TmpDistances tmp("tmp_distances", ts.size(), ts.size());
    SimplexLUT lut(ts.size() - (E - 1) * tau, top_k);
    knn(ts, ts, lut, tmp, E, tau, Tp, top_k);

    Dataset d_validation =
        load_csv("knn_test_validation_E" + std::to_string(E) + ".csv");

    const auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    const auto indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.indices);
    const auto validation =
        Kokkos::create_mirror_view_and_copy(HostSpace(), d_validation);

    CHECK(distances.extent(0) == validation.extent(0));
    CHECK(distances.extent(1) == validation.extent(1));

    CHECK(indices.extent(0) == validation.extent(0));
    CHECK(indices.extent(1) == validation.extent(1));

    for (size_t row = 0; row < distances.extent(0); row++) {
        for (size_t col = 0; col < distances.extent(1); col++) {
            CHECK(distances(row, col) == doctest::Approx(validation(row, col)));
        }
    }

    normalize_lut(lut);

    Kokkos::deep_copy(distances, lut.distances);

    for (size_t row = 0; row < distances.extent(0); row++) {
        auto sum = 0.0f;
        for (size_t col = 0; col < distances.extent(1); col++) {
            sum += distances(row, col);
        }

        CHECK(sum == doctest::Approx(1.0f));
    }
}

TEST_CASE("Compute kNN table for E=2") { test_knn_common(2); }

TEST_CASE("Compute kNN table for E=3") { test_knn_common(3); }

TEST_CASE("Compute kNN table for E=4") { test_knn_common(4); }

TEST_CASE("Compute kNN table for E=5") { test_knn_common(5); }

// Test for large datasets where n_pred * n_lib > INT_MAX
// This test verifies the fix for SIGSEGV with large 2D array indexing
//
// Reference (Apple M4, 10 cores):
//   - Memory usage: ~8.6 GB (TmpDistances: 46342^2 * 4 bytes)
//   - Execution time: ~90 seconds
//
// Run with: ./kedm-test --no-skip -tc="*large*"
TEST_CASE("Compute kNN table for large dataset" *
          doctest::skip(true)) // Skip by default due to memory/time requirements
{
    // n = 46342 gives n^2 = 2,147,578,964 > INT_MAX (2,147,483,647)
    const int n = 46342;
    const int E = 2;
    const int tau = 1;
    const int Tp = 0;
    const int top_k = E + 1;

    MutableTimeSeries ts("timeseries", n);
    Kokkos::parallel_for(
        "init_ts", n, KOKKOS_LAMBDA(int i) { ts(i) = static_cast<float>(i % 1000) / 1000.0f; });

    TmpDistances tmp("tmp_distances", n, n);
    SimplexLUT lut(n - (E - 1) * tau, top_k);

    knn(ts, ts, lut, tmp, E, tau, Tp, top_k);

    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);

    CHECK(distances.extent(0) == static_cast<size_t>(n - (E - 1) * tau));
    CHECK(distances.extent(1) == static_cast<size_t>(top_k));

    // Check that distances are non-negative (basic sanity)
    CHECK(distances(0, 0) >= 0.0f);
}

} // namespace edm
