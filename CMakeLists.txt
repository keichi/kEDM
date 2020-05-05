cmake_minimum_required(VERSION 3.14)

project(kokkos-edm)

add_compile_options(-Wall -Wextra -pedantic -Werror)

# Kokkos
find_package(Kokkos REQUIRED)

# Tests
enable_testing()

# doctest
add_subdirectory(src/thirdparty/doctest)
include(src/thirdparty/doctest/scripts/cmake/doctest.cmake)

# kNN kernel
add_executable(knn_test test/knn_test.cpp)
target_link_libraries(knn_test PRIVATE Kokkos::kokkos doctest::doctest)
doctest_discover_tests(knn_test WORKING_DIRECTORY ../test/)

# Simplex projection kernel
add_executable(simplex_test test/simplex_test.cpp)
target_link_libraries(simplex_test PRIVATE Kokkos::kokkos doctest::doctest)
doctest_discover_tests(simplex_test WORKING_DIRECTORY ../test/)
