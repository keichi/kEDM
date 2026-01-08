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
    TmpDistances valid_dist("valid_distances", N, L);
    TmpIndices valid_ind("valid_indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    Kokkos::deep_copy(valid_dist, dist);

    edm::partial_sort(dist, ind, K, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid_ind(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid_dist, i, Kokkos::ALL),
                Kokkos::subview(valid_ind, i, Kokkos::ALL));
        });

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_dist);
    auto valid_indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_ind);

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
    TmpDistances valid_dist("valid_distances", N, L);
    TmpIndices valid_ind("valid_indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    Kokkos::deep_copy(valid_dist, dist);

    edm::partial_sort_cpu(dist, ind, K, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid_ind(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid_dist, i, Kokkos::ALL),
                Kokkos::subview(valid_ind, i, Kokkos::ALL));
        });

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_dist);
    auto valid_indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_ind);

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

TEST_CASE("Full sort kNN LUT (CPU version)")
{
    int N = 100;
    int L = 1000;
    int n_partial = 1;
    int Tp = 1;

    Kokkos::Random_XorShift64_Pool<> random_pool(42);

    TmpDistances dist("distances", N, L);
    TmpIndices ind("indices", N, L);
    TmpDistances valid_dist("valid_distances", N, L);
    TmpIndices valid_ind("valid_indices", N, L);

    Kokkos::fill_random(dist, random_pool, 123456789.0f);

    Kokkos::deep_copy(valid_dist, dist);

    edm::full_sort_cpu(dist, ind, L, N, n_partial, Tp);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, L),
                [=](int j) { valid_ind(i, j) = j + n_partial + Tp; });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(valid_dist, i, Kokkos::ALL),
                Kokkos::subview(valid_ind, i, Kokkos::ALL));
        });

    auto distances = Kokkos::create_mirror_view_and_copy(HostSpace(), dist);
    auto indices = Kokkos::create_mirror_view_and_copy(HostSpace(), ind);

    auto valid_distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_dist);
    auto valid_indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_ind);

    for (int i = 0; i < distances.extent_int(0); i++) {
        for (int j = 0; j < L; j++) {
            CHECK(distances(i, j) ==
                  doctest::Approx(sqrt(valid_distances(i, j))));
            CHECK(indices(i, j) == valid_indices(i, j));
        }
    }
}

} // namespace edm
