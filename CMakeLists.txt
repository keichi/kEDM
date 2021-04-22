cmake_minimum_required(VERSION 3.10)

project(kokkos-edm)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

add_compile_options(-Wall -Wextra)
if(APPLE)
  # gcc needs this flag to compile Accelerate
  add_compile_options(-flax-vector-conversions)
endif()

# Kokkos
find_package(Kokkos REQUIRED)

# HDF5
find_package(HDF5 REQUIRED)

# MPI
option(USE_MPI "Enable MPI")
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND AND HDF5_IS_PARALLEL)
    add_executable(edm-xmap-mpi src/xmap_run_mpi.cpp)
    target_link_libraries(edm-xmap-mpi PRIVATE kedm Kokkos::kokkos HighFive
                          MPI::MPI_CXX)
  endif()
endif()

# HighFive
set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "")
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "")
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "")
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "")
add_subdirectory(src/thirdparty/highfive EXCLUDE_FROM_ALL)

# argh
add_subdirectory(src/thirdparty/argh EXCLUDE_FROM_ALL)

# Main library
add_library(kedm
  src/edim.cpp
  src/io.cpp
  src/simplex.cpp
  src/xmap.cpp
  src/knn.cpp
  src/smap.cpp
  src/stats.cpp)
target_link_libraries(kedm PRIVATE Kokkos::kokkos HighFive)
set_target_properties(kedm PROPERTIES POSITION_INDEPENDENT_CODE ON)

# LAPACK
if (Kokkos_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  target_link_libraries(kedm PRIVATE ${CUDA_cublas_LIBRARY})
else()
  find_package(LAPACK REQUIRED)
  target_link_libraries(kedm PRIVATE ${LAPACK_LIBRARIES})
endif()

# Executables
add_executable(edm-xmap src/xmap_run.cpp)
target_link_libraries(edm-xmap PRIVATE kedm Kokkos::kokkos argh HighFive)

add_executable(knn-bench src/knn_bench.cpp)
target_link_libraries(knn-bench PRIVATE kedm Kokkos::kokkos argh)

add_executable(lookup-bench src/lookup_bench.cpp)
target_link_libraries(lookup-bench PRIVATE kedm Kokkos::kokkos argh)

add_executable(smap-bench src/smap_bench.cpp)
target_link_libraries(smap-bench PRIVATE kedm Kokkos::kokkos argh)

option(USE_LIKWID "Enable LIKWID performance counters")
if(USE_LIKWID)
  find_package(likwid)

  target_link_libraries(knn-bench PRIVATE likwid::likwid)
  target_compile_definitions(knn-bench PRIVATE -DLIKWID_PERFMON)

  target_link_libraries(lookup-bench PRIVATE likwid::likwid)
  target_compile_definitions(lookup-bench PRIVATE -DLIKWID_PERFMON)
endif()

# Python bindings
option(USE_PYTHON "Enable Python bindings")
if(USE_PYTHON)
  add_subdirectory(src/thirdparty/pybind11 EXCLUDE_FROM_ALL)
  pybind11_add_module(_kedm src/bindings.cpp)
  target_link_libraries(_kedm PRIVATE kedm Kokkos::kokkos)
  install(TARGETS _kedm LIBRARY DESTINATION python/kedm)
endif()

# Tests
enable_testing()

# doctest
add_subdirectory(src/thirdparty/doctest EXCLUDE_FROM_ALL)
include(src/thirdparty/doctest/scripts/cmake/doctest.cmake)

# kNN
add_executable(knn_test test/knn_test.cpp)
target_link_libraries(knn_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

# Simplex projection
add_executable(simplex_test test/simplex_test.cpp)
target_link_libraries(simplex_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

# S-Map
add_executable(smap_test test/smap_test.cpp)
target_link_libraries(smap_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

# One-to-one cross mapping
add_executable(xmap_one_to_one_test test/xmap_one_to_one_test.cpp)
target_link_libraries(xmap_one_to_one_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

# All-to-all cross mapping
add_executable(xmap_all_to_all_test test/xmap_all_to_all_test.cpp)
target_link_libraries(xmap_all_to_all_test PRIVATE kedm Kokkos::kokkos HighFive doctest::doctest)

# Register tests
option(REGISTER_TESTS "Register unit tests" ON)

if(REGISTER_TESTS)
doctest_discover_tests(knn_test WORKING_DIRECTORY ../test/)
doctest_discover_tests(simplex_test WORKING_DIRECTORY ../test/)
doctest_discover_tests(smap_test WORKING_DIRECTORY ../test/)
doctest_discover_tests(xmap_one_to_one_test WORKING_DIRECTORY ../test/)
doctest_discover_tests(xmap_all_to_all_test WORKING_DIRECTORY ../test/)
endif()
