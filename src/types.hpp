#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>
namespace edm
{

#ifdef KOKKOS_ENABLE_CUDA
using DevSpace = Kokkos::CudaSpace;
#else
using DevSpace = Kokkos::DefaultExecutionSpace;
#endif
using HostSpace = Kokkos::HostSpace;

// Dataset is alyways left layout regardless of the backend
using Dataset = Kokkos::View<float **, Kokkos::LayoutLeft, DevSpace>;
// TimeSeries also has left layout even though it's 1D
using TimeSeries = Kokkos::View<float *, Kokkos::LayoutLeft, DevSpace>;

using ScratchTimeSeries = Kokkos::View<float *, DevSpace::scratch_memory_space,
                                       Kokkos::MemoryUnmanaged>;

using Distances = Kokkos::View<float **, DevSpace>;
using Indices = Kokkos::View<uint32_t **, DevSpace>;

using ScratchDistances = Kokkos::View<float *, DevSpace::scratch_memory_space,
                                      Kokkos::MemoryUnmanaged>;
using ScratchIndices = Kokkos::View<uint32_t *, DevSpace::scratch_memory_space,
                                    Kokkos::MemoryUnmanaged>;

struct LUT {
    Distances distances;
    Indices indices;

    LUT(uint32_t n_rows, uint32_t n_columns)
    {
        distances = Distances("distances", n_rows, n_columns);
        indices = Indices("indices", n_rows, n_columns);
    }
};

using CrossMap = Kokkos::View<float *, DevSpace>;

using HostTargets = Kokkos::View<uint32_t *, HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using DevTargets = Kokkos::View<uint32_t *, DevSpace>;

} // namespace edm

#endif
