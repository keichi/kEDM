#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Kokkos_Core.hpp>

namespace edm
{

using DevSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DevScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;
using HostSpace = Kokkos::HostSpace;

using Dataset = Kokkos::View<const float **, Kokkos::LayoutLeft, DevSpace>;
using MutableDataset = Kokkos::View<float **, Kokkos::LayoutLeft, DevSpace>;

using TimeSeries = Kokkos::View<const float *, Kokkos::LayoutLeft, DevSpace>;
using MutableTimeSeries = Kokkos::View<float *, Kokkos::LayoutLeft, DevSpace>;

// Used in lookup
using ScratchTimeSeries =
    Kokkos::View<float *, DevScratchSpace, Kokkos::MemoryUnmanaged>;

using Distances = Kokkos::View<float **, DevSpace>;
using Indices = Kokkos::View<int **, DevSpace>;

// Used in kNN (partial sort)
using TmpDistances = Kokkos::View<float **, Kokkos::LayoutRight, DevSpace>;
using ScratchDistances =
    Kokkos::View<float **, DevScratchSpace, Kokkos::MemoryUnmanaged>;
using ScratchIndices =
    Kokkos::View<int **, DevScratchSpace, Kokkos::MemoryUnmanaged>;

struct SimplexLUT {
    Distances distances;
    Indices indices;

    SimplexLUT(int n_rows, int n_columns)
    {
        distances = Distances("distances", n_rows, n_columns);
        indices = Indices("indices", n_rows, n_columns);
    }
};

using CrossMap = Kokkos::View<float *, DevSpace>;

using Targets = Kokkos::View<int *, DevSpace>;

} // namespace edm

#endif
