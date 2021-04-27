#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
yum install -y yum-utils hdf5-devel devtoolset-8

# Install CUDA 10.2
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel6/x86_64/cuda-rhel6.repo
yum install -y cuda-nvcc-10-2-10.2.89-1 cuda-cudart-dev-10-2-10.2.89-1 libcublas-10-2-10.2.3.254-1 libcublas-devel-10.2.2.89-1
ln -s /usr/local/cuda-10.2/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcuda.so.1
ln -s /usr/local/cuda-10.2 /usr/local/cuda

# Install Kokkos
git clone https://github.com/kokkos/kokkos.git
cd kokkos
git checkout 3.3.00
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_COMPILER=$(pwd)/../bin/nvcc_wrapper \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_VOLTA70=ON ..
make
make install
