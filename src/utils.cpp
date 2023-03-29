#include "utils.hpp"

#include <iomanip>

namespace edm
{

std::ostream &operator<<(std::ostream &os, const SimplexLUT &lut)
{
    os << "kNN LUT (" << lut.distances.extent(0) << "x"
       << lut.distances.extent(1) << ")" << std::endl;
    os << "========================================" << std::endl;

    os << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < lut.distances.extent(0); i++) {
        for (size_t j = 0; j < lut.indices.extent(1); j++) {
            os << lut.indices(i, j) << " ";
        }
        os << ": ";
        for (size_t j = 0; j < lut.distances.extent(1); j++) {
            os << lut.distances(i, j) << " ";
        }
        os << std::endl;
    }

    return os;
}

} // namespace edm
