Setting up
##########

Requirements
============

kEDM requires the following tools and libraries:

- `CMake <https://cmake.org/>`_ 3.10 or greater
- `Kokkos <https://github.com/kokkos/kokkos>`_ 3.2.00 or greater
- `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
- LAPACK (on CPU) or cuBLAS (on GPU)
- MPI (optional)

On Ubuntu, install the following packages via apt:

- CMake: ``cmake``
- HDF5: ``libhdf5-dev`` or ``libhdf5-openmpi-dev``
- LAPACK: ``libopenblas-dev`` and ``liblapacke-dev``
- MPI: ``libopenmpi-dev``

## Building Kokkos

Clone Kokkos using git. kEDM is tested with Kokkos 3.3.00 but might work with
other versions of Kokkos as well.

.. code-block:: bash

    $ git clone https://github.com/kokkos/kokkos.git
    $ cd kokkos
    $ git checkout 3.3.00

Then configure Kokkos using CMake and build. Kokkos provides a large number of
CMake options to control the backends and features to enable, which are
detailed in its `documentation <https://github.com/kokkos/kokkos/blob/master/BUILD.md>`_.
Below are minimal examples for configuring Kokkos.

Building for CPU
----------------

Change `-DCMAKE_CXX_COMPILER=<path/to/compiler>` to specify the C++ compiler
to use. `-DKokkos_ARCH_ZEN2=ON` should match with your CPU's micro
architecture.

.. code-block:: bash

    $ mkdir build-openmp
    $ cd build-openmp
    $ cmake \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-openmp \
      -DBUILD_SHARED_LIBS=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
      -DKokkos_ARCH_=ON ..
    $ make install

Building for GPU
----------------

Change ``-DKokkos_ARCH_VOLTA70=ON`` to match with your GPU's compute capability.
For example,

- NVIDIA A100: ``-DKokkos_ARCH_AMPERE80``
- NVIDIA V100: ``-DKokkos_ARCH_VOLTA70``
- GeForce RTX 3090: ``-DKokkos_ARCH_AMPERE86``
- GeForce RTX 2080: ``-DKokkos_ARCH_TURING75``

.. code-block:: bash

    $ mkdir build-cuda
    $ cd build-cuda
    $ cmake \
      -DCMAKE_CXX_COMPILER=$(pwd)/../bin/nvcc_wrapper \
      -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-cuda \
      -DBUILD_SHARED_LIBS=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DKokkos_ARCH_VOLTA70=ON ..
    $ make install

Building kEDM
=============

Clone kEDM using git. Note that the ``--recursive`` flag is required because
some third-party libraries are bundled as submodules.

.. code-block:: bash

    $ git clone --recursive https://github.com/keichi/kEDM.git
    $ cd kEDM

Then configure and build using CMake.

Building for CPU
----------------

Change ``-DCMAKE_CXX_COMPILER=<path/to/compiler>`` to specify the C++ compiler
to use.

.. code-block:: bash

    $ mkdir build-openmp
    $ cd build-openmp
    $ cmake \
      -DCMAKE_CXX_COMPILER=g++ \
      -DKokkos_DIR=$HOME/kokkos-openmp/lib/cmake/Kokkos \
      -DHIGHFIVE_USE_BOOST=OFF ..
    $ make

Building for GPU
----------------

.. code-block:: bash

    $ mkdir build-cuda
    $ cd build-cuda
    $ cmake
        -DCMAKE_CXX_COMPILER=$HOME/kokkos-cuda/bin/nvcc_wrapper \
        -DKokkos_DIR=$HOME/kokkos-cuda/lib/cmake/Kokkos \
        -DHIGHFIVE_USE_BOOST=OFF ..
    $ make

Testing kEDM
============

Run ``ctest`` within your build directory to execute the unit tests.

.. code-block:: bash

    $ ctest
