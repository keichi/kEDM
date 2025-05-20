#include <fstream>
#include <vector>

#ifdef HAVE_HDF5
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#endif

#include "io.hpp"

namespace edm
{

Dataset load_csv(const std::string &path)
{
    std::ifstream ifs(path);
    std::string line;
    std::vector<std::vector<float>> columns;

    if (!ifs) {
        throw std::invalid_argument("Failed to open file " + path);
    }

    auto n_rows = 0;
    auto n_columns = 0;
    auto is_header = true;

    while (ifs >> line) {
        std::stringstream ss(line);
        std::string cell;

        for (auto i = 0; std::getline(ss, cell, ','); i++) {
            // Read header
            if (is_header) {
                columns.push_back({});
                n_columns++;
                continue;
            }

            // Read body
            columns[i].push_back(std::stof(cell));
        }

        if (is_header) {
            is_header = false;
            continue;
        }
        n_rows++;
    }

    auto ds = MutableDataset("dataset", n_rows, n_columns);
    auto mirror = Kokkos::create_mirror_view(ds);

    for (auto i = 0; i < n_columns; i++) {
        for (auto j = 0; j < n_rows; j++) {
            mirror(j, i) = columns[i][j];
        }
    }

    Kokkos::deep_copy(ds, mirror);

    return Dataset(ds);
}

#ifdef HAVE_HDF5
Dataset load_hdf5(const HighFive::DataSet &dataset)
{
    const auto shape = dataset.getDimensions();

    const size_t n_rows = shape[0];
    const size_t n_columns = shape[1];
    const size_t MAX_CHUNK_SIZE = 100;

    std::vector<float> rows(MAX_CHUNK_SIZE * n_columns);

    auto ds = MutableDataset("dataset", n_rows, n_columns);
    auto mirror = Kokkos::create_mirror_view(ds);

    for (size_t i = 0; i < n_rows; i += MAX_CHUNK_SIZE) {
        const size_t chunk_size = std::min(MAX_CHUNK_SIZE, n_rows - i);

        dataset.select({i, 0}, {chunk_size, n_columns}).read_raw(rows.data());

        for (size_t j = 0; j < chunk_size; j++) {
            for (size_t k = 0; k < n_columns; k++) {
                mirror(i + j, k) = rows[j * n_columns + k];
            }
        }
    }

    Kokkos::deep_copy(ds, mirror);

    return Dataset(ds);
}
#endif

} // namespace edm
