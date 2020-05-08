cmake_minimum_required(VERSION 3.14)

project(kokkos-edm)

add_compile_options(-Wall -Wextra -pedantic -Werror)

# Kokkos
find_package(Kokkos REQUIRED)

# HDF5
find_package(HDF5 REQUIRED)
set(USE_BOOST OFF)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_EXAMPLES OFF)
add_subdirectory(src/thirdparty/highfive)

# Tests
enable_testing()

# doctest
add_subdirectory(src/thirdparty/doctest)
include(src/thirdparty/doctest/scripts/cmake/doctest.cmake)

# kNN
add_executable(knn_test test/knn_test.cpp)
target_link_libraries(knn_test PRIVATE Kokkos::kokkos HighFive doctest::doctest)
doctest_discover_tests(knn_test WORKING_DIRECTORY ../test/)

# Simplex projection
add_executable(simplex_test test/simplex_test.cpp)
target_link_libraries(simplex_test PRIVATE Kokkos::kokkos HighFive doctest::doctest)
doctest_discover_tests(simplex_test WORKING_DIRECTORY ../test/)

# One-to-one cross mapping
add_executable(xmap_one_to_one_test test/xmap_one_to_one_test.cpp)
target_link_libraries(xmap_one_to_one_test PRIVATE Kokkos::kokkos HighFive doctest::doctest)
doctest_discover_tests(xmap_one_to_one_test WORKING_DIRECTORY ../test/)

# All-to-all cross mapping
add_executable(xmap_all_to_all_test test/xmap_all_to_all_test.cpp)
target_link_libraries(xmap_all_to_all_test PRIVATE Kokkos::kokkos HighFive doctest::doctest)
doctest_discover_tests(xmap_all_to_all_test WORKING_DIRECTORY ../test/)
