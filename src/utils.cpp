#include "utils.hpp"

#include <iomanip>

namespace edm
{

std::ostream &operator<<(std::ostream &os, const SimplexLUT &lut)
{
    auto distances =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.distances);
    auto indices =
        Kokkos::create_mirror_view_and_copy(HostSpace(), lut.indices);

    os << "kNN LUT (" << distances.extent(0) << "x" << distances.extent(1)
       << ")" << std::endl;
    os << "========================================" << std::endl;

    os << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < distances.extent(0); i++) {
        for (size_t j = 0; j < indices.extent(1); j++) {
            os << indices(i, j) << " ";
        }
        os << ": ";
        for (size_t j = 0; j < distances.extent(1); j++) {
            os << distances(i, j) << " ";
        }
        os << std::endl;
    }

    return os;
}

} // namespace edm
