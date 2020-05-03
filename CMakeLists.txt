cmake_minimum_required(VERSION 3.14)

project(kokkos-edm)

add_executable(edm main.cpp)

find_package(Kokkos REQUIRED)

target_compile_options(edm PUBLIC -Wall -Wextra -pedantic -Werror)
target_link_libraries(edm Kokkos::kokkos)
