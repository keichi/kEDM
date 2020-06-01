# kokkos-edm

[![build](https://github.com/keichi/kEDM/workflows/build/badge.svg)](https://github.com/keichi/kEDM/actions?query=workflow%3Abuild)

An experimental implementation of Empirical Dynamic Modeling (EDM) using Kokkos

## Build

If Kokkos and HDF5 are installed under a non-standard path, `Kokkos_DIR` and
`HDF5_DIR` must be set appropriately.

### macOS

Use gcc since the default clang on macOS does not support OpenMP.

```
cmake -DCMAKE_C_COMPILER=$(which gcc-9) \
      -DCMAKE_CXX_COMPILER=$(which g++-9) \
      -DCMAKE_BUILD_TYPE=Debug ..
```

### Linux (with CUDA)

```
cmake -DCMAKE_CXX_COMPILER=/opt/kokkos/bin/nvcc_wrapper ..
```
