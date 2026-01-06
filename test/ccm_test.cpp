#include <cfloat>

#include <Kokkos_NestedSort.hpp>
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

TEST_CASE("Partially sort kNN LUT with Radix-based Selection")
{
    int N = 100;
    int L = 1000;
    int K = 123;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    SimplexLUT lut(N, L);
    SimplexLUT valid(N, L);

    Kokkos::fill_random(lut.distances, random_pool, 123456789.0f);

    Kokkos::deep_copy(valid.distances, lut.distances);

    edm::partial_sort_radix(lut, K, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid.indices(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid.distances, i, Kokkos::ALL),
                Kokkos::subview(valid.indices, i, Kokkos::ALL));
        });

    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    auto indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.indices);

    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid.distances);
    auto valid_indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid.indices);

    for (int i = 0; i < distances.extent_int(0); i++) {
        for (int j = 0; j < K; j++) {
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_distances(i, j))));
            CHECK(indices(i, j) == valid_indices(i, j));
        }
        for (int j = K; j < L; j++) {
            CHECK(distances(i, j) == FLT_MAX);
            CHECK(indices(i, j) == -1);
        }
    }
}

TEST_CASE("Partially sort kNN LUT with Bitonic Top-K")
{
    int N = 100;
    int L = 1000;
    int K = 123;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    SimplexLUT lut(N, L);
    SimplexLUT valid(N, L);

    Kokkos::fill_random(lut.distances, random_pool, 123456789.0f);

    Kokkos::deep_copy(valid.distances, lut.distances);

    edm::partial_sort_bitonic(lut, K, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid.indices(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid.distances, i, Kokkos::ALL),
                Kokkos::subview(valid.indices, i, Kokkos::ALL));
        });

    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    auto indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.indices);

    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid.distances);
    auto valid_indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid.indices);

    for (int i = 0; i < distances.extent_int(0); i++) {
        for (int j = 0; j < K; j++) {
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_distances(i, j))));
            CHECK(indices(i, j) == valid_indices(i, j));
        }
        for (int j = K; j < L; j++) {
            CHECK(distances(i, j) == FLT_MAX);
            CHECK(indices(i, j) == -1);
        }
    }
}

TEST_CASE("Bitonic Top-K with k=1")
{
    int n_partial = 1;
    int Tp = 1;
    int N = 50;
    int L = 100;
    int K = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    SimplexLUT lut(N, L);
    SimplexLUT valid(N, L);

    Kokkos::fill_random(lut.distances, random_pool, 123456789.0f);
    Kokkos::deep_copy(valid.distances, lut.distances);

    edm::partial_sort_bitonic(lut, K, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid.indices(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid.distances, i, Kokkos::ALL),
                Kokkos::subview(valid.indices, i, Kokkos::ALL));
        });

    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid.distances);

    for (int i = 0; i < N; i++) {
        CHECK(distances(i, 0) ==
              doctest::Approx(sqrt(valid_distances(i, 0))));
    }
}

TEST_CASE("Bitonic Top-K with large k values")
{
    int n_partial = 1;
    int Tp = 1;
    int N = 100;
    int L = 5000;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    std::vector<int> k_values = {256, 512, 1024, 2048};

    for (int K : k_values) {
        if (K >= L) continue;

        CAPTURE(K);

        SimplexLUT lut(N, L);
        SimplexLUT valid(N, L);

        Kokkos::fill_random(lut.distances, random_pool, 123456789.0f);
        Kokkos::deep_copy(valid.distances, lut.distances);

        edm::partial_sort_bitonic(lut, K, L, N, n_partial, Tp);

        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
                int i = member.league_rank();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, L),
                    [=](int j) { valid.indices(i, j) = j + n_partial + Tp; });

                member.team_barrier();

                Kokkos::Experimental::sort_by_key_team(
                    member, Kokkos::subview(valid.distances, i, Kokkos::ALL),
                    Kokkos::subview(valid.indices, i, Kokkos::ALL));
            });

        auto distances =
            Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
        auto valid_distances =
            Kokkos::create_mirror_view_and_copy(HostSpace(), valid.distances);

        // Check first few elements match
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < std::min(K, 10); j++) {
                CHECK(distances(i, j) ==
                      doctest::Approx(sqrt(valid_distances(i, j))));
            }
        }
    }
}

}
