# AGENTS.md

This document provides guidance for AI assistants working on the kEDM codebase.

## Project Overview

**kEDM (Kokkos-EDM)** is a high-performance C++ implementation of the Empirical Dynamical Modeling
(EDM) framework for analyzing time series data. It provides optimized, parallelized implementations
of EDM algorithms for CPUs and GPUs.

### Key Algorithms
- **Simplex projection** - Nonlinear time series forecasting
- **S-Map** - Sequential Locally Weighted Global Linear Maps
- **CCM** - Convergent Cross Mapping for causal inference

### Technology Stack
- **Language**: C++17
- **Parallelization**: Kokkos (supports OpenMP for CPU, CUDA for GPU)
- **Build System**: CMake (minimum 3.16)
- **Python Bindings**: pybind11

## Project Structure

```
kEDM/
├── src/           # Core C++ implementation
├── test/          # C++ unit tests (doctest)
├── python/        # Python package and tests
├── docs/          # Sphinx documentation
└── scripts/       # Utility scripts
```

## Build Instructions

### Configure and Build

```bash
cmake -B build -S .
cmake --build build
```

### Common CMake Options

| Flag | Description | Default |
|------|-------------|---------|
| `-DKEDM_ENABLE_CPU=ON/OFF` | Enable CPU backend (OpenMP) | ON |
| `-DKEDM_ENABLE_GPU=ON/OFF` | Enable GPU backend (CUDA) | OFF |
| `-DKEDM_ENABLE_TESTS=ON/OFF` | Build unit tests | ON |
| `-DKEDM_ENABLE_PYTHON=ON/OFF` | Build Python bindings | OFF |
| `-DKEDM_ENABLE_MPI=ON/OFF` | Enable MPI support | OFF |

## Running Tests

### C++ Tests

```bash
cd build
ctest
```

Or run the test executable directly:

```bash
OMP_PROC_BIND=false ./kedm-test
```

### Python Tests

```bash
pytest python/tests
```

## Code Style

This project uses **clang-format** for C++ code formatting. The configuration is in `.clang-format`.

**Before committing, always format your code:**

```bash
clang-format -i src/*.cpp src/*.hpp test/*.cpp
```

The CI pipeline enforces formatting. PRs with unformatted code will fail checks.

## Architecture Notes

### Core Types (`src/types.hpp`)

The project uses Kokkos Views for memory management:

- `Dataset` / `MutableDataset` - 2D arrays for time series collections
- `TimeSeries` / `MutableTimeSeries` - 1D arrays for time series
- `SimplexLUT` - Lookup tables for k-NN distances and indices

### Important Considerations

1. **Integer Types for Large Data**: Use `size_t` instead of `int` for loop indices when iterating
   over data that could exceed `INT_MAX`. This was a recent bug fix (see commit `3249100`).

2. **Kokkos Patterns**: All parallel code uses Kokkos abstractions (`Kokkos::parallel_for`,
   `Kokkos::parallel_reduce`, etc.) for CPU/GPU portability.

3. **Memory Spaces**: Be aware of host vs. device memory when working with GPU code.

## Dependencies

Dependencies are automatically fetched via CMake FetchContent:

- **Kokkos** 5.0.0 - Parallel computing framework
- **HighFive** - HDF5 C++ interface
- **doctest** - Testing framework
- **pybind11** - Python bindings
- **Boost Math** - Mathematical functions
- **argh** - Command-line parsing

## CI/CD

GitHub Actions runs on every push/PR:

1. **Format check** - Validates clang-format compliance
2. **Build & Test** - Multiple platforms:
   - Linux: gcc-12, clang-14, CUDA 12.4
   - macOS: Clang, GCC-12

## Creating a New Release

1. Update version numbers in `pyproject.toml` and `python/kedm/__init__.py`.
2. Ensure all tests pass (`ctest` and `pytest`)
3. Run `clang-format` on all source files
4. Commit the version bump
5. Create a git tag: `git tag vX.Y.Z`
6. Push the tag: `git push origin vX.Y.Z`
