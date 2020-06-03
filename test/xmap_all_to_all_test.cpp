#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "../src/edim.hpp"
#include "../src/io.hpp"
#include "../src/types.hpp"
#include "../src/xmap.hpp"

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
        TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edim(ts, 20, 1, 1);

        CHECK(optimal_E[i] == optimal_E_valid[i]);
    }

    std::vector<Targets> groups;
    group_ts(groups, optimal_E, 20);

    CrossMap rho("xmap", ds.extent(1));
    std::vector<float> rho_valid(ds.extent(1));

    for (auto i = 0u; i < ds.extent(1); i++) {
        TimeSeries library(ds, Kokkos::ALL, i);

        xmap(rho, ds, library, groups, 20, 1, 0);

        ds_corrcoef.select({i, 0}, {1, ds.extent(1)}).read(rho_valid);

        auto rho_mirror = Kokkos::create_mirror_view_and_copy(HostSpace(), rho);

        for (auto j = 0u; j < ds.extent(1); j++) {
            CHECK(rho_mirror[j] == doctest::Approx(rho_valid[j]));
        }
    }
}

TEST_CASE("Compute all-to-all cross mappings") { xmap_test_common(); }

} // namespace edm

int main(int argc, char **argv)
{
    Kokkos::initialize();

    int res = doctest::Context(argc, argv).run();

    Kokkos::finalize();

    return res;
}
