#!/bin/bash

LIBOMP_VER=15.0.0
LIBOMP_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-$LIBOMP_VER/openmp-$LIBOMP_VER.src.tar.xz
CMAKE_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-$LIBOMP_VER/cmake-$LIBOMP_VER.src.tar.xz

cd $(mktemp -d)

mkdir {src,cmake}
curl -sL $LIBOMP_URL | tar xf - --strip-components 1 -C src
curl -sL $CMAKE_URL | tar xf - --strip-components 1 -C cmake

cmake -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_INSTALL_PREFIX=/opt/homebrew/opt/libomp -S src -B build
cmake --build build -j $(sysctl -n hw.physicalcpu_max) 
sudo cmake --install build
