#include <cfloat>

#include <Kokkos_Random.hpp>
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

TEST_CASE("Compute Convergent Cross Mapping (naive algorithm)")
{
    const int E = 3;
    const int tau = 1;
    const int Tp = 0;
    const int sample = 100;

    std::vector<int> lib_sizes;
    for (int i = 10; i <= 75; i += 5) {
        lib_sizes.push_back(i);
    }

    const Dataset ds = load_csv("sardine_anchovy_sst.csv");
    const auto anchovy = Kokkos::subview(ds, Kokkos::ALL, 1);
    const auto sst = Kokkos::subview(ds, Kokkos::ALL, 4);

    const auto rhos_naive1 =
        ccm_naive(anchovy, sst, lib_sizes, sample, E, tau, Tp, 42);
    const auto rhos_naive2 =
        ccm_naive(sst, anchovy, lib_sizes, sample, E, tau, Tp, 42);
    const auto rhos_opt1 = ccm(anchovy, sst, lib_sizes, sample, E, tau, Tp, 42);
    const auto rhos_opt2 = ccm(sst, anchovy, lib_sizes, sample, E, tau, Tp, 42);

    for (size_t i = 0; i < rhos_naive1.size(); i++) {
        CHECK(rhos_naive1[i] == doctest::Approx(rhos_opt1[i]).epsilon(0.05));
        CHECK(rhos_naive2[i] == doctest::Approx(rhos_opt2[i]).epsilon(0.05));
    }
}

TEST_CASE("Partially sort kNN LUT")
{
    int N = 100;
    int L = 1000;
    int K = 123;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    TmpDistances dist("distances", N, L);
    TmpIndices ind("indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    // Create a deep copy of dist before sorting
    auto valid_dist = Kokkos::create_mirror(HostSpace(), dist);
    Kokkos::deep_copy(valid_dist, dist);

    edm::partial_sort(dist, ind, K, L, N, n_partial, Tp);

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            // Check that index points to correct original distance
            int orig_idx = indices(i, j) - n_partial - Tp;
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_dist(i, orig_idx))));

            // Check that distances are sorted in ascending order
            if (j > 0) {
                CHECK(distances(i, j - 1) <= distances(i, j));
            }
        }
        for (int j = K; j < L; j++) {
            CHECK(distances(i, j) == FLT_MAX);
            CHECK(indices(i, j) == -1);
        }
    }
}

TEST_CASE("Partially sort kNN LUT (CPU version)")
{
    int N = 100;
    int L = 1000;
    int K = 123;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    TmpDistances dist("distances", N, L);
    TmpIndices ind("indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    // Create a deep copy of dist before sorting
    auto valid_dist = Kokkos::create_mirror(HostSpace(), dist);
    Kokkos::deep_copy(valid_dist, dist);

    edm::partial_sort_cpu(dist, ind, K, L, N, n_partial, Tp);

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            // Check that index points to correct original distance
            int orig_idx = indices(i, j) - n_partial - Tp;
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_dist(i, orig_idx))));

            // Check that distances are sorted in ascending order
            if (j > 0) {
                CHECK(distances(i, j - 1) <= distances(i, j));
            }
        }
        for (int j = K; j < L; j++) {
            CHECK(distances(i, j) == FLT_MAX);
            CHECK(indices(i, j) == -1);
        }
    }
}

TEST_CASE("Full sort kNN LUT")
{
    int N = 100;
    int L = 1000;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    TmpDistances dist("distances", N, L);
    TmpIndices ind("indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    // Create a deep copy of dist before sorting
    auto valid_dist = Kokkos::create_mirror(HostSpace(), dist);
    Kokkos::deep_copy(valid_dist, dist);

    edm::full_sort(dist, ind, L, N, n_partial, Tp);

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < L; j++) {
            // Check that index points to correct original distance
            int orig_idx = indices(i, j) - n_partial - Tp;
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_dist(i, orig_idx))));

            // Check that distances are sorted in ascending order
            if (j > 0) {
                CHECK(distances(i, j - 1) <= distances(i, j));
            }
        }
    }
}

TEST_CASE("Full sort kNN LUT (CPU version)")
{
    int N = 100;
    int L = 1000;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    TmpDistances dist("distances", N, L);
    TmpIndices ind("indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    // Create a deep copy of dist before sorting
    auto valid_dist = Kokkos::create_mirror(HostSpace(), dist);
    Kokkos::deep_copy(valid_dist, dist);

    edm::full_sort_cpu(dist, ind, L, N, n_partial, Tp);

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < L; j++) {
            // Check that index points to correct original distance
            int orig_idx = indices(i, j) - n_partial - Tp;
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_dist(i, orig_idx))));

            // Check that distances are sorted in ascending order
            if (j > 0) {
                CHECK(distances(i, j - 1) <= distances(i, j));
            }
        }
    }
}

} // namespace edm
