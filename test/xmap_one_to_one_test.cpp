#include <doctest/doctest.h>

#include "../src/io.hpp"
#include "../src/simplex.hpp"
#include "../src/types.hpp"

namespace edm
{

// Validation data was generated with pyEDM 1.0.1 with the following
// parameters:
// pyEDM.Simplex(dataFrame=pyEDM.sampleData["sardine_anchovy_sst"],
//               E=3, Tp=1, columns="anchovy", target="np_sst", lib="1 76",
//               pred="1 76", verbose=True)
void cross_mapping_test_common(int E)
{
    const auto tau = 1;
    const auto Tp = 1;

    Dataset ds1 = load_csv("sardine_anchovy_sst.csv");
    Dataset ds2 =
        load_csv("anchovy_sst_validation_E" + std::to_string(E) + ".csv");

    const auto library = TimeSeries(
        ds1, std::make_pair<size_t, size_t>(0, ds1.extent(0) - (E - 1)), 1);
    const auto target = TimeSeries(
        ds1, std::make_pair<size_t, size_t>(0, ds1.extent(0) - (E - 1)), 4);
    const auto valid_prediction = TimeSeries(ds2, Kokkos::ALL, 0);

    MutableTimeSeries prediction("prediction", target.size() - (E - 1) * tau);
    simplex(prediction, library, library, target, E, tau, Tp);

    const auto pred =
        Kokkos::create_mirror_view_and_copy(HostSpace(), prediction);
    const auto valid =
        Kokkos::create_mirror_view_and_copy(HostSpace(), valid_prediction);

    CHECK(pred.size() == valid.size());

    for (size_t i = 0; i < pred.size(); i++) {
        CHECK(pred(i) == doctest::Approx(valid(i)));
    }
}

TEST_CASE("Compute one-to-one cross mapping for E=2")
{
    cross_mapping_test_common(2);
}

TEST_CASE("Compute one-to-one cross mapping for E=3")
{
    cross_mapping_test_common(3);
}

TEST_CASE("Compute one-to-one cross mapping for E=4")
{
    cross_mapping_test_common(4);
}

TEST_CASE("Compute one-to-one cross mapping for E=5")
{
    cross_mapping_test_common(5);
}

} // namespace edm
