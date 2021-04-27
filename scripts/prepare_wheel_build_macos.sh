#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Need unreleased version of delocate
# https://github.com/matthew-brett/delocate/pull/76
pip install --force-reinstall git+https://github.com/matthew-brett/delocate.git@52b9185ff6ebb643392286242ea06417143dadb7

# Install dependencies
brew install hdf5

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
sudo make install
