#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "../src/edim.hpp"
#include "../src/io.hpp"
#include "../src/stats.hpp"
#include "../src/types.hpp"
#include "../src/xmap.hpp"

void run(const std::string &input_path, const std::string &dataset,
         const std::string &output_path)
{
    const int E_max = 20;
    const int tau = 1;

    HighFive::File output(
        output_path, HighFive::File::Overwrite,
        HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

    const auto ds = edm::load_hdf5(input_path, dataset);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = ds.extent(1) / size;
    int block_start = block_size * rank;
    int block_end =
        std::min(block_size * (rank + 1), static_cast<int>(ds.extent(1)));

    std::vector<int> optimal_E(block_size * size);

    for (auto i = block_start; i < block_end; i++) {
        Kokkos::Timer timer;

        edm::TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edm::edim(ts, E_max, tau, 1);

        std::cout << "Simpelx for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;
    }

    MPI_Allgather(MPI_IN_PLACE, block_size, MPI_INT, optimal_E.data(),
                  block_size, MPI_INT, MPI_COMM_WORLD);

    auto dataspace = HighFive::DataSpace({ds.extent(1)});
    auto ds_edim = output.createDataSet<int>("/embedding", dataspace);
    ds_edim.write(optimal_E);

    std::vector<edm::LUT> luts;

    // Allocate kNN tables
    for (int E = 1; E <= E_max; E++) {
        luts.push_back(edm::LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    edm::TmpDistances tmp("tmp_distances", ds.extent(0), ds.extent(0));

    std::vector<edm::Targets> groups;
    edm::group_ts(groups, optimal_E, E_max);

    edm::CrossMap ccm("ccm", ds.extent(1));
    edm::CrossMap rho("rho", ds.extent(1));

    auto ccm_mirror = Kokkos::create_mirror_view(ccm);
    auto rho_mirror = Kokkos::create_mirror_view(rho);

    dataspace = HighFive::DataSpace({ds.extent(1), ds.extent(1)});
    auto ds_ccm = output.createDataSet<float>("/ccm", dataspace);
    auto ds_rho = output.createDataSet<float>("/rho", dataspace);

    for (auto i = block_start; i < block_end; i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(ccm, ds, library, groups, luts, tmp, E_max, tau, 0);

        std::cout << "Cross map for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;

        Kokkos::deep_copy(ccm_mirror, ccm);
        ds_ccm.select({i, 0}, {1, ds.extent(1)}).write(ccm_mirror.data());
    }

    for (auto i = block_start; i < block_end; i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::corrcoef(rho, ds, library);

        std::cout << "Correlation for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;

        Kokkos::deep_copy(rho_mirror, rho);
        ds_rho.select({i, 0}, {1, ds.extent(1)}).write(rho_mirror.data());
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (!rank) {
            std::cout << "Usage: edm-xmap input output dataset" << std::endl;
        }
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string dataset = argv[3];

    Kokkos::initialize();

    run(input_path, dataset, output_path);

    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}
