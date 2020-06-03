#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "../src/edim.hpp"
#include "../src/io.hpp"
#include "../src/types.hpp"
#include "../src/xmap.hpp"

void run(const std::string &path, const std::string &dataset)
{
    const uint32_t E_max = 20;
    const int32_t tau = 1;

    const auto ds = edm::load_hdf5(path, dataset);

    std::vector<uint32_t> optimal_E(ds.extent(1));

    for (auto i = 0u; i < ds.extent(1); i++) {
        Kokkos::Timer timer;

        edm::TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edm::edim(ts, E_max, tau, 1);

        std::cout << "Simpelx for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;
    }

    std::vector<edm::LUT> luts;

    // Allocate kNN tables
    for (uint32_t E = 1; E <= E_max; E++) {
        luts.push_back(edm::LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    edm::LUT tmp_lut(ds.extent(0), ds.extent(0));

    std::vector<edm::Targets> groups;
    edm::group_ts(groups, optimal_E, E_max);

    edm::CrossMap rho("xmap", ds.extent(1));

    for (auto i = 0u; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(rho, ds, library, groups, luts, tmp_lut, E_max, tau, 0);

        std::cout << "Cross map for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "Usage: edm-xmap file dataset" << std::endl;
        return 1;
    }
    const std::string path = argv[1];
    const std::string dataset = argv[2];

    Kokkos::initialize();

    run(path, dataset);

    Kokkos::finalize();

    return 0;
}
