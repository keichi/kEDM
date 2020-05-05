#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/simplex.hpp"
#include "../src/types.hpp"

void simplex_test_common(uint32_t E)
{
    const uint32_t tau = 1;
    const auto Tp = 1;

    edm::Dataset ds1 = edm::load_csv("simplex_test_data.csv");
    edm::Dataset ds2 =
        edm::load_csv("simplex_test_validation_E" + std::to_string(E) + ".csv");

    edm::TimeSeries ts(ds1, Kokkos::ALL, 0);
    edm::TimeSeries library(ts, std::make_pair(0ul, ts.size() / 2));
    edm::TimeSeries target(ts, std::make_pair(ts.size() / 2 - (E - 1) * tau,
                                              ts.size() - (E - 1) * tau));

    edm::TimeSeries valid_prediction(ds2, Kokkos::ALL, 0);

    edm::LUT cache(target.size(), library.size());
    edm::NearestNeighbors knn(cache);

    edm::LUT lut(target.size() - (E - 1) * tau, E + 1);
    knn.run(library, target, lut, E, tau, Tp, E + 1);
    edm::normalize_lut(lut);

    edm::TimeSeries prediction("prediction", target.size() - (E - 1) * tau);
    edm::simplex(prediction, library, lut);

    for (size_t i = 0; i < prediction.size(); i++) {
        CHECK(prediction(i) ==
              doctest::Approx(valid_prediction(i)).epsilon(0.01));
    }
}

TEST_CASE("Compute simplex projection for E=2")
{
    Kokkos::initialize();

    simplex_test_common(2);

    Kokkos::finalize();
}

TEST_CASE("Compute simplex projection for E=3")
{
    Kokkos::initialize();

    simplex_test_common(3);

    Kokkos::finalize();
}

TEST_CASE("Compute simplex projection for E=4")
{
    Kokkos::initialize();

    simplex_test_common(4);

    Kokkos::finalize();
}

TEST_CASE("Compute simplex projection for E=5")
{
    Kokkos::initialize();

    simplex_test_common(5);

    Kokkos::finalize();
}

// Test data is generated using pyEDM with the following parameters:
// pyEDM.EmbedDimension(dataFrame=pyEDM.sampleData["TentMap"], lib="1 100",
//                      pred="201 500", columns="TentMap", target="TentMap",
//                      maxE=20)
// template <class T, class U> void embed_dim_test_common()
// {
//     const auto tau = 1;
//     const auto Tp = 1;
//     const auto max_E = 20;
//
//     DataFrame df1, df2;
//     df1.load_csv("TentMap_rEDM.csv");
//     df2.load_csv("TentMap_rEDM_validation.csv");
//
//     auto knn = std::unique_ptr<NearestNeighbors>(new T(tau, Tp, true));
//     auto simplex = std::unique_ptr<Simplex>(new U(tau, Tp, true));
//
//     std::vector<float> buffer;
//     LUT lut;
//     std::vector<float> rho(max_E);
//     std::vector<float> rho_valid(max_E);
//
//     for (auto E = 1; E <= max_E; E++) {
//         const auto ts = df1.columns[1];
//         const auto library = ts.slice(0, 100);
//         const auto target = ts.slice(200 - (E - 1) * tau, 500);
//
//         knn->compute_lut(lut, library, target, E);
//         lut.normalize();
//
//         const auto prediction = simplex->predict(buffer, lut, library, E);
//         const auto shifted_target = simplex->shift_target(target, E);
//
//         rho[E - 1] = corrcoef(prediction, shifted_target);
//         rho_valid[E - 1] = df2.columns[1][E - 1];
//
//         // Check correlation coefficient
//         REQUIRE(rho[E - 1] == Approx(rho_valid[E - 1]));
//     }
//
//     const auto it = std::max_element(rho.begin(), rho.end());
//     const auto it2 = std::max_element(rho_valid.begin(), rho_valid.end());
//
//     // Check optimal embedding dimension
//     REQUIRE(it - rho.begin() == it2 - rho_valid.begin());
// }
//
// TEST_CASE("Find optimal embedding dimension (CPU)", "[simplex][cpu]")
// {
//     embed_dim_test_common<NearestNeighborsCPU, SimplexCPU>();
// }
