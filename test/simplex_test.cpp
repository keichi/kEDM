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
    MutableTimeSeries prediction("prediction", target.size() - (E - 1) * tau);

    simplex(prediction, library, target, E, tau, Tp);

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

// Test data is generated using pyEDM 1.14.0.0 as follows:
//
// df = pd.read_csv("test/block_3sp.csv")
// S = EDM.Simplex(dataFrame = df,
//                 lib = "1 99", pred = "100 198", E = 3, Tp = 1,
//                 tau = -1, columns = "x_t y_t z_t", target = "x_t")
// S["Predictions"].dropna().to_csv("block_3sp_validation.csv", index=False)
TEST_CASE("Compute multivariate simplex projection for E=3")
{
    const int E = 3;
    const int tau = 1;
    const int Tp = 1;

    const Dataset ds1 = load_csv("block_3sp.csv");
    const Dataset ds2 = load_csv("block_3sp_validation.csv");

    const MutableDataset library("library", 99, 3);
    const MutableDataset target("target", 101, 3);

    Kokkos::deep_copy(
        Kokkos::subview(library, Kokkos::ALL, 0),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(0, 99), 1));
    Kokkos::deep_copy(
        Kokkos::subview(library, Kokkos::ALL, 1),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(0, 99), 4));
    Kokkos::deep_copy(
        Kokkos::subview(library, Kokkos::ALL, 2),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(0, 99), 7));

    Kokkos::deep_copy(
        Kokkos::subview(target, Kokkos::ALL, 0),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(97, 198), 1));
    Kokkos::deep_copy(
        Kokkos::subview(target, Kokkos::ALL, 1),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(97, 198), 4));
    Kokkos::deep_copy(
        Kokkos::subview(target, Kokkos::ALL, 2),
        Kokkos::subview(ds1, std::make_pair<size_t, size_t>(97, 198), 7));

    const MutableDataset prediction("prediction", 99, 3);

    simplex(prediction, library, target, E, tau, Tp);

    const auto pred = Kokkos::create_mirror_view_and_copy(
        HostSpace(), Kokkos::subview(prediction, Kokkos::ALL, 0));
    const auto valid = Kokkos::create_mirror_view_and_copy(
        HostSpace(), Kokkos::subview(ds2, Kokkos::ALL, 0));

    for (size_t i = 0; i < pred.size(); i++) {
        CHECK(pred(i) == doctest::Approx(valid(i)));
    }
}

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

    TmpDistances tmp("tmp_distances", 400, 100);

    for (auto E = 1; E <= E_max; E++) {
        TimeSeries ts(ds1, Kokkos::ALL, 1);
        TimeSeries library(ts, std::make_pair(0, 100));
        TimeSeries target(ts, std::make_pair(200 - (E - 1) * tau, 500));

        SimplexLUT lut(target.size() - (E - 1) * tau, E + 1);
        knn(library, target, lut, tmp, E, tau, Tp, E + 1);
        normalize_lut(lut);

        MutableTimeSeries prediction("prediction",
                                     target.size() - (E - 1) * tau);
        TimeSeries shifted_target(
            target,
            std::make_pair<size_t, size_t>((E - 1) * tau + Tp, target.size()));

        lookup(prediction, library, lut);

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
