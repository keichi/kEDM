#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>

namespace edm
{

using DevSpace = Kokkos::DefaultExecutionSpace;
using HostSpace = Kokkos::HostSpace;

// Dataset is alyways left layout regardless of the backend
using Dataset = Kokkos::View<float **, Kokkos::LayoutLeft, DevSpace>;
// TimeSeries also has left layout even though it's 1D
using TimeSeries = Kokkos::View<float *, Kokkos::LayoutLeft, DevSpace>;

struct LUT {
    Kokkos::View<float **, DevSpace> distances;
    Kokkos::View<uint32_t **, DevSpace> indices;

    LUT(uint32_t n_rows, uint32_t n_columns)
    {
        distances =
            Kokkos::View<float **, DevSpace>("distances", n_rows, n_columns);
        indices =
            Kokkos::View<uint32_t **, DevSpace>("indices", n_rows, n_columns);
    }
};

} // namespace edm

#endif
