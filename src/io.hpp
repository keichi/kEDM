#ifndef __IO_HPP__
#define __IO_HPP__

#include <fstream>
#include <vector>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include "types.hpp"

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

    auto ds = Dataset("dataset", n_rows, n_columns);
    auto mirror = Kokkos::create_mirror_view(ds);

    for (auto i = 0; i < n_columns; i++) {
        for (auto j = 0; j < n_rows; j++) {
            mirror(j, i) = columns[i][j];
        }
    }

    Kokkos::deep_copy(ds, mirror);

    return ds;
}

Dataset load_hdf5(const std::string &path, const std::string &ds_name)
{
    const auto file = HighFive::File(path, HighFive::File::ReadOnly);
    const auto dataset = file.getDataSet(ds_name);
    const auto shape = dataset.getDimensions();

    size_t n_rows = shape[0];
    size_t n_columns = shape[1];

    const size_t MAX_CHUNK_SIZE = 100;

    std::vector<float> rows(MAX_CHUNK_SIZE * n_columns);

    auto ds = Dataset("dataset", n_rows, n_columns);
    auto mirror = Kokkos::create_mirror_view(ds);

    for (auto i = 0u; i < n_rows; i += MAX_CHUNK_SIZE) {
        const auto chunk_size = std::min(MAX_CHUNK_SIZE, n_rows - i);

        dataset.select({i, 0u}, {chunk_size, n_columns}).read(rows.data());

        for (auto j = 0u; j < chunk_size; j++) {
            for (auto k = 0u; k < n_columns; k++) {
                ds(i + j, k) = rows[j * n_columns + k];
            }
        }
    }

    return ds;
}

} // namespace edm

#endif
