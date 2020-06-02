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
    const auto ds = edm::load_hdf5(path, dataset);

    std::vector<uint32_t> optimal_E(ds.extent(1));

    for (auto i = 0u; i < ds.extent(1); i++) {
        Kokkos::Timer timer;

        edm::TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edm::edim(ts, 20, 1, 1);

        std::cout << "Simpelx for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;
    }

    edm::CrossMap rho("xmap", ds.extent(1));

    for (auto i = 0u; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(rho, ds, library, optimal_E, 20, 1, 0);

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
