#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/simplex.hpp"
#include "../src/types.hpp"

namespace edm
{

// Validation data was generated with pyEDM 1.0.1 with the following
// parameters:
// pyEDM.Simplex(dataFrame=pyEDM.sampleData["sardine_anchovy_sst"],
//               E=3, Tp=1, columns="anchovy", target="np_sst", lib="1 76",
//               pred="1 76", verbose=True)

void cross_mapping_test_common(uint32_t E)
{
    const auto tau = 1;
    const auto Tp = 1;

    Dataset ds1 = load_csv("sardine_anchovy_sst.csv");
    Dataset ds2 =
        load_csv("anchovy_sst_validation_E" + std::to_string(E) + ".csv");

    LUT cache(ds1.extent(0), ds1.extent(0));
    NearestNeighbors knn(cache);

    const auto library =
        TimeSeries(ds1, std::make_pair(0ul, ds1.extent(0) - (E - 1)), 1u);
    const auto target =
        TimeSeries(ds1, std::make_pair(0ul, ds1.extent(0) - (E - 1)), 4u);
    const auto valid_prediction = TimeSeries(ds2, Kokkos::ALL, 0);

    LUT lut(target.size() - (E - 1) * tau, E + 1);
    knn.run(library, library, lut, E, tau, Tp, E + 1);
    normalize_lut(lut);

    TimeSeries prediction("prediction", target.size() - (E - 1) * tau);
    simplex(prediction, target, lut);

    CHECK(prediction.size() == valid_prediction.size());

    for (auto i = 0u; i < prediction.size(); i++) {
        CHECK(prediction(i) == doctest::Approx(valid_prediction(i)));
    }
}

TEST_CASE("Compute one-to-one cross mapping for E=2")
{
    Kokkos::initialize();

    cross_mapping_test_common(2);

    Kokkos::finalize();
}

TEST_CASE("Compute one-to-one cross mapping for E=3")
{
    Kokkos::initialize();

    cross_mapping_test_common(3);

    Kokkos::finalize();
}

TEST_CASE("Compute one-to-one cross mapping for E=4")
{
    Kokkos::initialize();

    cross_mapping_test_common(4);

    Kokkos::finalize();
}

TEST_CASE("Compute one-to-one cross mapping for E=5")
{
    Kokkos::initialize();

    cross_mapping_test_common(5);

    Kokkos::finalize();
}

} // namespace edm
