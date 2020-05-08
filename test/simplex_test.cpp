#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cmath>
#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/simplex.hpp"
#include "../src/types.hpp"

namespace edm
{

void simplex_test_common(uint32_t E)
{
    const uint32_t tau = 1;
    const auto Tp = 1;

    Dataset ds1 = load_csv("simplex_test_data.csv");
    Dataset ds2 =
        load_csv("simplex_test_validation_E" + std::to_string(E) + ".csv");

    TimeSeries ts(ds1, Kokkos::ALL, 0);
    TimeSeries library(ts, std::make_pair(0ul, ts.size() / 2));
    TimeSeries target(ts, std::make_pair(ts.size() / 2 - (E - 1) * tau,
                                         ts.size() - (E - 1) * tau));

    TimeSeries valid_prediction(ds2, Kokkos::ALL, 0);

    LUT cache(target.size(), library.size());
    NearestNeighbors knn(cache);

    LUT lut(target.size() - (E - 1) * tau, E + 1);
    knn.run(library, target, lut, E, tau, Tp, E + 1);
    normalize_lut(lut);

    TimeSeries prediction("prediction", target.size() - (E - 1) * tau);
    simplex(prediction, library, lut);

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

float corrcoef(const TimeSeries x, const TimeSeries y)
{
    const auto n = std::min(x.size(), y.size());
    auto mean_x = 0.0f, mean_y = 0.0f;
    auto sum_xy = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f;

    for (auto i = 0u; i < n; i++) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    for (auto i = 0u; i < n; i++) {
        auto diff_x = x[i] - mean_x;
        auto diff_y = y[i] - mean_y;

        sum_xy += diff_x * diff_y;
        sum_x2 += diff_x * diff_x;
        sum_y2 += diff_y * diff_y;
    }

    return sum_xy / std::sqrt(sum_x2 * sum_y2);
}

// Test data is generated using pyEDM with the following parameters:
// pyEDM.EmbedDimension(dataFrame=pyEDM.sampleData["TentMap"], lib="1 100",
//                      pred="201 500", columns="TentMap", target="TentMap",
//                      maxE=20)
void embed_dim_test_common()
{
    const size_t tau = 1;
    const auto Tp = 1;
    const auto E_max = 20;

    Dataset ds1 = load_csv("TentMap_rEDM.csv");
    Dataset ds2 = load_csv("TentMap_rEDM_validation.csv");

    std::vector<float> rho(E_max);
    std::vector<float> rho_valid(E_max);

    LUT cache(400, 100);

    for (auto E = 1; E <= E_max; E++) {
        TimeSeries ts(ds1, Kokkos::ALL(), 1);
        TimeSeries library(ts, std::make_pair(0ul, 100ul));
        TimeSeries target(ts, std::make_pair(200ul - (E - 1) * tau, 500ul));

        LUT lut(target.size() - (E - 1) * tau, E + 1);

        NearestNeighbors knn(cache);
        knn.run(library, target, lut, E, tau, Tp, E + 1);
        normalize_lut(lut);

        for (auto i = 0u; i < lut.distances.extent(0); i++) {
            for (auto j = 0u; j < lut.distances.extent(1); j++) {
                std::cout << lut.distances(i, j) << " ";
            }

            std::cout << std::endl;
        }

        TimeSeries prediction("prediction", target.size() - (E - 1) * tau - Tp);
        TimeSeries shifted_target(
            target, std::make_pair((E - 1) * tau + Tp, target.size()));

        simplex(prediction, library, lut);

        rho[E - 1] = corrcoef(prediction, shifted_target);
        rho_valid[E - 1] = ds2(E - 1, 1);

        // Check correlation coefficient
        CHECK(rho[E - 1] == doctest::Approx(rho_valid[E - 1]));
    }

    const auto it = std::max_element(rho.begin(), rho.end());
    const auto it2 = std::max_element(rho_valid.begin(), rho_valid.end());

    // Check optimal embedding dimension
    CHECK(it - rho.begin() == it2 - rho_valid.begin());
}

TEST_CASE("Compute optimal embedding dimension")
{
    Kokkos::initialize();

    embed_dim_test_common();

    Kokkos::finalize();
}

} // namespace edm
