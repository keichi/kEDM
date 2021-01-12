#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "edim.hpp"
#include "io.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "xmap.hpp"

const int E_max = 20;
const int tau = 1;

void run_find_embedding_dimension(edm::Dataset ds, std::vector<int> &optimal_E,
                                  HighFive::File &output)
{
    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;

        edm::TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edm::edim(ts, E_max, tau, 1);

        std::cout << "Simpelx for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;
    }

    auto dataspace = HighFive::DataSpace::From(optimal_E);
    auto ds_edim = output.createDataSet<int>("/embedding", dataspace);
    ds_edim.write(optimal_E);
}

void run_convergent_cross_mapping(edm::Dataset ds,
                                  const std::vector<int> &optimal_E,
                                  HighFive::File &output)
{
    std::vector<edm::LUT> luts;

    for (int E = 1; E <= E_max; E++) {
        luts.push_back(edm::LUT(ds.extent(0) - (E - 1) * tau, E + 1));
    }

    edm::TmpDistances tmp("tmp_distances", ds.extent(0), ds.extent(0));

    std::vector<edm::Targets> groups;
    edm::group_ts(groups, optimal_E, E_max);

    edm::CrossMap ccm("ccm", ds.extent(1));
    auto ccm_mirror = Kokkos::create_mirror_view(ccm);

    auto dataspace = HighFive::DataSpace({ds.extent(1), ds.extent(1)});
    auto ds_ccm = output.createDataSet<float>("/ccm", dataspace);

    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(ccm, ds, library, groups, luts, tmp, E_max, tau, 0);

        Kokkos::deep_copy(ccm_mirror, ccm);

        std::cout << "Cross map for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;

        ds_ccm.select({i, 0}, {1, ds.extent(1)}).write(ccm_mirror.data());
    }
}

void run_cross_correlation(edm::Dataset ds, HighFive::File &output)
{
    edm::CrossMap rho("rho", ds.extent(1));
    auto rho_mirror = Kokkos::create_mirror_view(rho);

    auto dataspace = HighFive::DataSpace({ds.extent(1), ds.extent(1)});
    auto ds_rho = output.createDataSet<float>("/rho", dataspace);

    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::corrcoef(rho, ds, library);

        Kokkos::deep_copy(rho_mirror, rho);

        std::cout << "Correlation for time series #" << i << " took "
                  << timer.seconds() << " seconds." << std::endl;

        ds_rho.select({i, 0}, {1, ds.extent(1)}).write(rho_mirror.data());
    }
}

void run(const std::string &input_path, const std::string &dataset,
         const std::string &output_path)
{
    HighFive::File output(output_path, HighFive::File::Overwrite);

    const auto ds = edm::load_hdf5(input_path, dataset);

    std::vector<int> optimal_E(ds.extent(1));

    run_find_embedding_dimension(ds, optimal_E, output);
    run_convergent_cross_mapping(ds, optimal_E, output);
    run_cross_correlation(ds, output);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cout << "Usage: edm-xmap input output dataset" << std::endl;
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string dataset = argv[3];

    Kokkos::initialize();

    run(input_path, dataset, output_path);

    Kokkos::finalize();

    return 0;
}
