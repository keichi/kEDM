cmake_minimum_required(VERSION 3.10)

project(kokkos-edm)

add_compile_options(-Wall -Wextra -pedantic -Wno-unknown-pragmas)

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
add_subdirectory(src/thirdparty/highfive)

# argh
add_subdirectory(src/thirdparty/argh)

add_library(kedm SHARED
  src/edim.cpp
  src/io.cpp
  src/simplex.cpp
  src/xmap.cpp
  src/knn.cpp
  src/smap.cpp
  src/stats.cpp)
target_link_libraries(kedm PRIVATE Kokkos::kokkos HighFive)

# LAPACK
if (Kokkos_ENABLE_CUDA)
  find_package(CUDA)
else()
  find_package(LAPACK)
endif()

if (Kokkos_ENABLE_CUDA)
  target_link_libraries(kedm PRIVATE ${CUDA_cublas_LIBRARY})
else()
  target_link_libraries(kedm PRIVATE ${LAPACK_LIBRARIES})
endif()

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

# Tests
enable_testing()

# doctest
add_subdirectory(src/thirdparty/doctest)
include(src/thirdparty/doctest/scripts/cmake/doctest.cmake)

# kNN
add_executable(knn_test test/knn_test.cpp)
target_link_libraries(knn_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

# Simplex projection
add_executable(simplex_test test/simplex_test.cpp)
target_link_libraries(simplex_test PRIVATE kedm Kokkos::kokkos doctest::doctest)

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
doctest_discover_tests(xmap_one_to_one_test WORKING_DIRECTORY ../test/)
doctest_discover_tests(xmap_all_to_all_test WORKING_DIRECTORY ../test/)
endif()
