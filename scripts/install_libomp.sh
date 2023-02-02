#!/bin/bash

LIBOMP_VER=13.0.0

export CXXFLAGS=""
export LDFLAGS=""

cd $(mktemp -d)

curl -sL https://github.com/llvm/llvm-project/releases/download/llvmorg-$LIBOMP_VER/openmp-$LIBOMP_VER.src.tar.xz | tar xvf -
cd openmp-$LIBOMP_VER.src

cmake -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" .
make -j $(sysctl -n hw.physicalcpu_max)
sudo make install
