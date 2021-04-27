#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
brew install hdf5

# Install Kokkos
git clone https://github.com/kokkos/kokkos.git
cd kokkos
git checkout 3.4.00
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON ..
make
sudo make install
