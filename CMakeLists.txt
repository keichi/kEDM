cmake_minimum_required(VERSION 3.22)

# Allow building with dependencies that specify cmake_minimum_required < 3.5
# (e.g., argh v1.3.2 specifies VERSION 3.1)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(kokkos-edm CXX)

include(FetchContent)

# Compiler flags
add_compile_options(-Wall -Wextra)
if(APPLE)
  # gcc needs this flag to compile Accelerate
  add_compile_options(-flax-vector-conversions)
endif()

if(LINUX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "")
endif()

# LTO is unsporrted by CUDA
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Kokkos
option(KEDM_ENABLE_CPU "Enable CPU (OpenMP) backend" ON)
if(KEDM_ENABLE_CPU)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "")
endif()
option(KEDM_ENABLE_GPU "Enable GPU (CUDA) backend" OFF)
if(KEDM_ENABLE_GPU)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
endif()

FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        5.0.0
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(kokkos)

# argh
FetchContent_Declare(
  argh
  GIT_REPOSITORY https://github.com/adishavit/argh.git
  GIT_TAG        v1.3.2
)
FetchContent_MakeAvailable(argh)

# Main library
add_library(kedm
  src/ccm.cpp
  src/edim.cpp
  src/io.cpp
  src/simplex.cpp
  src/xmap.cpp
  src/knn.cpp
  src/smap.cpp
  src/stats.cpp)
target_link_libraries(kedm PRIVATE Kokkos::kokkos)

# boost-math
FetchContent_Declare(
  boost-math
  GIT_REPOSITORY https://github.com/boostorg/math.git
  GIT_TAG        boost-1.89.0
)
FetchContent_MakeAvailable(boost-math)
target_link_libraries(kedm PRIVATE Boost::math)

# pcg
FetchContent_Declare(
  pcg
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
  GIT_TAG        v0.98.1
)
FetchContent_MakeAvailable(pcg)
target_include_directories(kedm PRIVATE ${pcg_SOURCE_DIR}/include)

# LAPACK
if (Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(kedm PRIVATE CUDA::cublas)
else()
  find_package(LAPACK REQUIRED)
  target_link_libraries(kedm PRIVATE ${LAPACK_LIBRARIES})
endif()

# MPI
option(KEDM_ENABLE_MPI "Enable MPI")
if(KEDM_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# Executables
option(KEDM_ENABLE_EXECUTABLES "Build executables" ON)
if(KEDM_ENABLE_EXECUTABLES)
  # HighFive
  set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "")
  set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "")
  set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "")
  set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "")

  FetchContent_Declare(
    highfive
    GIT_REPOSITORY https://github.com/highfive-devs/highfive.git
    GIT_TAG        v3.2.0
  )
  FetchContent_MakeAvailable(highfive)

  target_link_libraries(kedm PRIVATE HighFive)
  target_compile_definitions(kedm PRIVATE -DHAVE_HDF5)

  add_executable(edm-xmap src/xmap_run.cpp)
  target_link_libraries(edm-xmap PRIVATE kedm Kokkos::kokkos argh HighFive)
  target_compile_definitions(edm-xmap PRIVATE -DHAVE_HDF5)

  add_executable(knn-bench src/knn_bench.cpp)
  target_link_libraries(knn-bench PRIVATE kedm Kokkos::kokkos argh)

  add_executable(lookup-bench src/lookup_bench.cpp)
  target_link_libraries(lookup-bench PRIVATE kedm Kokkos::kokkos argh)

  add_executable(smap-bench src/smap_bench.cpp)
  target_link_libraries(smap-bench PRIVATE kedm Kokkos::kokkos argh)

  if(MPI_CXX_FOUND AND HDF5_IS_PARALLEL)
    add_executable(edm-xmap-mpi src/xmap_run_mpi.cpp)
    target_link_libraries(edm-xmap-mpi PRIVATE kedm Kokkos::kokkos HighFive
                          MPI::MPI_CXX)
    target_compile_definitions(edm-xmap-mpi PRIVATE -DHAVE_HDF5)
  endif()
endif()

option(KEDM_ENABLE_LIKWID "Enable LIKWID performance counters")
if(KEDM_ENABLE_LIKWID)
  find_package(likwid)

  target_link_libraries(knn-bench PRIVATE likwid::likwid)
  target_compile_definitions(knn-bench PRIVATE -DLIKWID_PERFMON)

  target_link_libraries(lookup-bench PRIVATE likwid::likwid)
  target_compile_definitions(lookup-bench PRIVATE -DLIKWID_PERFMON)
endif()

option(KEDM_ENABLE_SCRATCH_MEMORY "Use Kokkos scratch memory" ON)
if(KEDM_ENABLE_SCRATCH_MEMORY)
  target_compile_definitions(kedm PRIVATE -DUSE_SCRATCH_MEMORY)
endif()

option(KEDM_ENABLE_SIMD_PRIMITIVES "Use Kokkos SIMD primitives" ON)
if(KEDM_ENABLE_SIMD_PRIMITIVES)
  target_compile_definitions(kedm PRIVATE -DUSE_SIMD_PRIMITIVES)
endif()

# Python bindings
option(KEDM_ENABLE_PYTHON "Build Python bindings")
if(KEDM_ENABLE_PYTHON)
  # pybind11
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.1
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(_kedm src/bindings.cpp)
  target_link_libraries(_kedm PRIVATE kedm Kokkos::kokkos)
  install(TARGETS _kedm DESTINATION .)
endif()

# Unit tests
option(KEDM_ENABLE_TESTS "Build unit tests" ON)
if(KEDM_ENABLE_TESTS)
  enable_testing()

  # doctest
  FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12
  )
  FetchContent_MakeAvailable(doctest)
  include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

  # Test executable
  add_executable(kedm-test
    test/main.cpp
    test/ccm_test.cpp
    test/knn_test.cpp
    test/simplex_test.cpp
    test/smap_test.cpp
    test/stats_test.cpp
    test/xmap_one_to_one_test.cpp
    test/xmap_all_to_all_test.cpp
  )
  target_link_libraries(kedm-test PRIVATE kedm Kokkos::kokkos doctest::doctest)

  doctest_discover_tests(kedm-test WORKING_DIRECTORY ../test/
                         PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false"
                         ADD_LABELS 1)
endif()

message(STATUS "kEDM build configuration:")
message(STATUS "  C++ Compiler:     ${CMAKE_CXX_COMPILER_ID} "
                                   "${CMAKE_CXX_COMPILER_VERSION} "
                                   "${CMAKE_CXX_COMPILER_WRAPPER}")
message(STATUS "                    ${CMAKE_CXX_COMPILER}")
message(STATUS "  CPU backend:      ${KEDM_ENABLE_CPU}")
message(STATUS "  GPU backend:      ${KEDM_ENABLE_GPU}")
message(STATUS "  Executables:      ${KEDM_ENABLE_EXECUTABLES}")
message(STATUS "  Python bindings:  ${KEDM_ENABLE_PYTHON}")
message(STATUS "  MPI:              ${KEDM_ENABLE_MPI}")
message(STATUS "  Unit tests:       ${KEDM_ENABLE_TESTS}")
message(STATUS "  LIKWID:           ${KEDM_ENABLE_LIKWID}")
message(STATUS "  Scratch memory:   ${KEDM_ENABLE_SCRATCH_MEMORY}")
message(STATUS "  SIMD primitives:  ${KEDM_ENABLE_SIMD_PRIMITIVES}")
