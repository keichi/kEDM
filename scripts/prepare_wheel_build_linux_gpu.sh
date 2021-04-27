#!/bin/bash

set -ex

# Install build dependencies
pip install scikit-build cmake

# Install dependencies
yum install -y yum-utils hdf5-devel

# Install CUDA 11.3
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum install -y cuda-nvcc-11-3-11.3.58-1 cuda-cudart-devel-11-3-11.3.58-1 libcublas-11-3-11.4.2.10064-1 libcublas-devel-11-3-11.4.2.10064-1 cuda-drivers-465.19.01-1
ln -s /usr/local/cuda-11.3 /usr/local/cuda

# Install Kokkos
git clone https://github.com/kokkos/kokkos.git
cd kokkos
git checkout 3.4.00
curl -O https://gist.githubusercontent.com/keichi/f12abe7ff807de1d6b47c9847d363b66/raw/7755aa53ddbcec9b25c691cb00949d23c67a29e1/cuda_multi_arch.diff
patch -p1 < cuda_multi_arch.diff
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_COMPILER=$(pwd)/../bin/nvcc_wrapper \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_PASCAL60=ON \
  -DKokkos_ARCH_PASCAL61=ON \
  -DKokkos_ARCH_VOLTA70=ON \
  -DKokkos_ARCH_VOLTA72=ON \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ARCH_AMPERE86=ON ..
make
make install
