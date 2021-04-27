#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
yum install -y openblas-devel hdf5-devel

# Install Kokkos
git clone https://github.com/kokkos/kokkos.git
cd kokkos
git checkout 3.3.00
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON ..
make
make install
