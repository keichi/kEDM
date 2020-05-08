#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest/doctest.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "../src/edim.hpp"
#include "../src/io.hpp"
#include "../src/knn.hpp"
#include "../src/simplex.hpp"
#include "../src/types.hpp"

namespace edm
{

void xmap_test_common()
{
    const auto ds = load_hdf5("xmap_all_to_all_test.h5", "values");

    const HighFive::File file("xmap_all_to_all_test_validation.h5");
    const auto ds_corrcoef = file.getDataSet("corrcoef");
    const auto ds_edim = file.getDataSet("embedding");

    std::vector<uint32_t> optimal_E(ds.extent(1));
    std::vector<uint32_t> optimal_E_valid(ds.extent(1));

    ds_edim.read(optimal_E_valid);

    for (auto i = 0u; i < ds.extent(1); i++) {
        optimal_E[i] = edim(TimeSeries(ds, Kokkos::ALL, i), 20, 1, 1);

        CHECK(optimal_E[i] == optimal_E_valid[i]);
    }

    std::vector<float> rhos(ds.extent(1));
    std::vector<float> rhos_valid(ds.extent(1));

    // for (auto i = 0u; i < ds.extent(1); i++) {
    //     xmap(rhos, ds, i, optimal_E);
    //     ds_corrcoef.select({i, 0}, {1, ds.extent(1)}).read(rhos_valid);

    //     for (auto j = 0u; j < ds.extent(1); j++) {
    //         CHECK(rhos[j] == Approx(rhos_valid[j]).margin(1e-5));
    //     }
    // }
}

TEST_CASE("Compute all-to-all cross mappings")
{
    Kokkos::initialize();

    xmap_test_common();

    Kokkos::finalize();
}

} // namespace edm
