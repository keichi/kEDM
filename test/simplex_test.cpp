#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/simplex.hpp"
#include "../src/stats.hpp"
#include "../src/types.hpp"

namespace edm
{

void simplex_test_common(int E)
{
    const int tau = 1;
    const int Tp = 1;

    const Dataset ds1 = load_csv("simplex_test_data.csv");
    const Dataset ds2 =
        load_csv("simplex_test_validation_E" + std::to_string(E) + ".csv");

    const TimeSeries ts(ds1, Kokkos::ALL, 0);
    const TimeSeries library(ts,
                             std::make_pair<size_t, size_t>(0, ts.size() / 2));
    const TimeSeries target(ts, std::make_pair(ts.size() / 2 - (E - 1) * tau,
                                               ts.size() - (E - 1) * tau));

    TimeSeries valid_prediction(ds2, Kokkos::ALL, 0);

    LUT tmp_lut(target.size(), library.size());
    LUT lut(target.size() - (E - 1) * tau, E + 1);
    knn(library, target, lut, tmp_lut, E, tau, Tp, E + 1);
    normalize_lut(lut);

    MutableTimeSeries prediction("prediction", target.size() - (E - 1) * tau);
    simplex(prediction, library, lut);

    const auto pred =
        Kokkos::create_mirror_view_and_copy(HostSpace(), prediction);
    const auto valid =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_prediction);

    for (size_t i = 0; i < pred.size(); i++) {
        CHECK(pred(i) == doctest::Approx(valid(i)).epsilon(0.01));
    }
}

TEST_CASE("Compute simplex projection for E=2") { simplex_test_common(2); }

TEST_CASE("Compute simplex projection for E=3") { simplex_test_common(3); }

TEST_CASE("Compute simplex projection for E=4") { simplex_test_common(4); }

TEST_CASE("Compute simplex projection for E=5") { simplex_test_common(5); }

// Test data is generated using pyEDM with the following parameters:
// pyEDM.EmbedDimension(dataFrame=pyEDM.sampleData["TentMap"], lib="1 100",
//                      pred="201 500", columns="TentMap", target="TentMap",
//                      maxE=20)
void embed_dim_test_common()
{
    const int tau = 1;
    const int Tp = 1;
    const int E_max = 20;

    Dataset ds1 = load_csv("TentMap_rEDM.csv");
    Dataset ds2 = load_csv("TentMap_rEDM_validation.csv");

    auto ds2_mirror = Kokkos::create_mirror_view_and_copy(HostSpace(), ds2);

    std::vector<float> rho(E_max);
    std::vector<float> rho_valid(E_max);

    LUT tmp_lut(400, 100);

    for (auto E = 1; E <= E_max; E++) {
        TimeSeries ts(ds1, Kokkos::ALL, 1);
        TimeSeries library(ts, std::make_pair(0, 100));
        TimeSeries target(ts, std::make_pair(200 - (E - 1) * tau, 500));

        LUT lut(target.size() - (E - 1) * tau, E + 1);
        knn(library, target, lut, tmp_lut, E, tau, Tp, E + 1);
        normalize_lut(lut);

        MutableTimeSeries prediction("prediction",
                                     target.size() - (E - 1) * tau);
        TimeSeries shifted_target(
            target,
            std::make_pair<size_t, size_t>((E - 1) * tau + Tp, target.size()));

        simplex(prediction, library, lut);

        rho[E - 1] = corrcoef(prediction, shifted_target);
        rho_valid[E - 1] = ds2_mirror(E - 1, 1);

        // Check correlation coefficient
        CHECK(rho[E - 1] == doctest::Approx(rho_valid[E - 1]));
    }

    const auto it = std::max_element(rho.begin(), rho.end());
    const auto it2 = std::max_element(rho_valid.begin(), rho_valid.end());

    // Check optimal embedding dimension
    CHECK(it - rho.begin() == it2 - rho_valid.begin());
}

TEST_CASE("Compute optimal embedding dimension") { embed_dim_test_common(); }

} // namespace edm

int main(int argc, char **argv)
{
    Kokkos::initialize();

    int res = doctest::Context(argc, argv).run();

    Kokkos::finalize();

    return res;
}
