#include <iostream>
#include <string>
#include <vector>

#include <argh.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "edim.hpp"
#include "io.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "xmap.hpp"

typedef struct {
    std::string input_path;
    std::string output_path;
    std::string dataset;
    int E_max;
    int tau;
    bool cross_correlation;
    bool verbose;
} Config;

Config config;

void run_find_embedding_dimension(edm::Dataset ds, std::vector<int> &optimal_E,
                                  HighFive::File &output)
{
    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;

        edm::TimeSeries ts(ds, Kokkos::ALL, i);
        optimal_E[i] = edm::edim(ts, config.E_max, config.tau, 1);

        if (config.verbose) {
            std::cout << "Simpelx for time series #" << i << " took "
                      << timer.seconds() << " seconds." << std::endl;
        }
    }

    auto dataspace = HighFive::DataSpace::From(optimal_E);
    auto ds_edim = output.createDataSet<int>("/embedding_dimension", dataspace);
    ds_edim.write(optimal_E);
}

void run_convergent_cross_mapping(edm::Dataset ds,
                                  const std::vector<int> &optimal_E,
                                  HighFive::File &output)
{
    std::vector<edm::LUT> luts;

    for (int E = 1; E <= config.E_max; E++) {
        luts.push_back(edm::LUT(ds.extent(0) - (E - 1) * config.tau, E + 1));
    }

    edm::TmpDistances tmp("tmp_distances", ds.extent(0), ds.extent(0));

    std::vector<edm::Targets> groups;
    edm::group_ts(groups, optimal_E, config.E_max);

    edm::CrossMap ccm("ccm", ds.extent(1));
    auto ccm_mirror = Kokkos::create_mirror_view(ccm);

    auto dataspace = HighFive::DataSpace({ds.extent(1), ds.extent(1)});
    auto ds_ccm = output.createDataSet<float>("/ccm_rho", dataspace);

    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::xmap(ccm, ds, library, groups, luts, tmp, config.E_max, config.tau,
                  0);

        Kokkos::deep_copy(ccm_mirror, ccm);

        if (config.verbose) {
            std::cout << "Cross map for time series #" << i << " took "
                      << timer.seconds() << " seconds." << std::endl;
        }

        ds_ccm.select({i, 0}, {1, ds.extent(1)}).write(ccm_mirror.data());
    }
}

void run_cross_correlation(edm::Dataset ds, HighFive::File &output)
{
    edm::CrossMap rho("rho", ds.extent(1));
    auto rho_mirror = Kokkos::create_mirror_view(rho);

    auto dataspace = HighFive::DataSpace({ds.extent(1), ds.extent(1)});
    auto ds_rho = output.createDataSet<float>("/cross_correlation", dataspace);

    for (size_t i = 0; i < ds.extent(1); i++) {
        Kokkos::Timer timer;
        edm::TimeSeries library(ds, Kokkos::ALL, i);

        edm::corrcoef(rho, ds, library);

        Kokkos::deep_copy(rho_mirror, rho);

        if (config.verbose) {
            std::cout << "Cross correlation for time series #" << i << " took "
                      << timer.seconds() << " seconds." << std::endl;
        }

        ds_rho.select({i, 0}, {1, ds.extent(1)}).write(rho_mirror.data());
    }
}

void run()
{
    HighFive::File output(config.output_path, HighFive::File::Overwrite);

    const auto ds = edm::load_hdf5(config.input_path, config.dataset);

    std::vector<int> optimal_E(ds.extent(1));

    run_find_embedding_dimension(ds, optimal_E, output);
    run_convergent_cross_mapping(ds, optimal_E, output);

    if (config.cross_correlation) {
        run_cross_correlation(ds, output);
    }
}

// clang-format off
void usage(const std::string &app_name)
{
    std::string msg =
        app_name +
        ": All-to-all Convergent Cross Mapping Application\n"
        "\n"
        "Usage:\n"
        "  " +
        app_name +
        " [OPTION...] input output\n"
        "  -d, --dataset arg           HDF5 dataset name (default: \"values\")\n"
        "  -e, --max-embedding-dim arg Embedding dimension (default: 20)\n"
        "  -t, --tau arg               Time delay (default: 1)\n"
        "  --cross-correlation         Compute cross correlation (default: false)\n"
        "  -h, --help                  Show this help";

    std::cout << msg << std::endl;
}
// clang-format on

int main(int argc, char *argv[])
{
    argh::parser cmdl(
        {"-d", "--dataset", "-e", "--max-embedding-dim", "-t", "--tau"});
    cmdl.parse(argc, argv);

    if (cmdl[{"-h", "--help"}]) {
        usage(argv[0]);
        return 0;
    }

    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    cmdl(1) >> config.input_path;
    cmdl(2) >> config.output_path;
    cmdl({"-d", "--dataset"}, "values") >> config.dataset;
    cmdl({"e", "embedding-dim"}, 20) >> config.E_max;
    cmdl({"t", "tau"}, 1) >> config.tau;

    config.cross_correlation = cmdl[{"--cross-correlation"}];
    config.verbose = cmdl[{"-v", "--verbose"}];

    Kokkos::initialize();

    run();

    Kokkos::finalize();

    return 0;
}
