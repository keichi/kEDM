#include <doctest/doctest.h>

#include "../src/edim.hpp"
#include "../src/io.hpp"
#include "../src/types.hpp"
#include "../src/xmap.hpp"

namespace edm
{

void xmap_test_common()
{
    const int E_max = 20;
    const int tau = 1;

    const auto ds = load_csv("xmap_all_to_all_test_input.csv");
    const auto ds_rho = load_csv("xmap_all_to_all_test_validation_rho.csv");
    const auto ds_edim = load_csv("xmap_all_to_all_test_validation_e.csv");

    const auto rho_valid_mirror =
        Kokkos::create_mirror_view_and_copy(HostSpace(), ds_rho);
    const auto edim_valid_mirror =
        Kokkos::create_mirror_view_and_copy(HostSpace(), ds_edim);

    std::vector<int> optimal_E(ds.extent(1));

    for (size_t i = 0; i < ds.extent(1); i++) {
        TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edim(ts, E_max, 1, 1);

        CHECK(optimal_E[i] == edim_valid_mirror(i, 0));
    }

    std::vector<SimplexLUT> luts;

    // Allocate kNN tables
    for (int E = 1; E <= E_max; E++) {
        luts.push_back(SimplexLUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    TmpDistances tmp("tmp_distances", ds.extent(0), ds.extent(0));

    std::vector<Targets> groups;
    group_ts(groups, optimal_E, E_max);

    CrossMap rho("xmap", ds.extent(1));
    std::vector<float> rho_valid(ds.extent(1));

    for (size_t i = 0; i < ds.extent(1); i++) {
        TimeSeries library(ds, Kokkos::ALL, i);

        xmap(rho, ds, library, groups, luts, tmp, E_max, 1, 0);

        auto rho_mirror = Kokkos::create_mirror_view_and_copy(HostSpace(), rho);

        for (size_t j = 0; j < ds.extent(1); j++) {
            CHECK(rho_mirror(j) == doctest::Approx(rho_valid_mirror(i, j)));
        }
    }
}

TEST_CASE("Compute all-to-all cross mappings") { xmap_test_common(); }

} // namespace edm
