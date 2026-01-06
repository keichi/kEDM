Install using pip
#################

Prebuilt wheels
---------------

The easiest way to install kEDM is to use the prebuilt wheels. Prebuilt wheels
for Linux (x86_64) and macOS (x86_64 and arm64) are available on PyPI. GPU
wheels are distributed as separate packages, one for each CUDA major version.

Requirements
============

- Python >= 3.10
- pip >= 21.0

Installation
============

CPU (Linux and macOS):

.. code-block:: bash

    $ pip3 install kedm

NVIDIA GPU (CUDA 12):

.. code-block:: bash

    $ pip3 install kedm-cuda12x

NVIDIA GPU (CUDA 13):

.. code-block:: bash

    $ pip3 install kedm-cuda13x

.. _pip-source:

Source
------

If you want to run on GPUs or use a specific compiler or LAPACK library, you
need to install from source.

Requirements
============

- Python >= 3.10
- pip >= 21.0
- A C++ compiler supporting C++20 (currently tested on GCC and Clang)
- LAPACK (on CPU)
- CUDA Toolkit (on GPU, currently tested on CUDA >= 12)

On Ubuntu, install the following packages via apt:

.. code-block:: bash

    $ sudo apt-get install git g++ libopenblas-serial-dev

On macOS, install the following packages via Homebrew:

.. code-block:: bash

    $ brew install libomp

Installation
============

(Optional) To target GPUs, set the ``CMAKE_ARGS`` environment variable as
follows:

.. code-block:: bash

    $ export CMAKE_ARGS="-DKEDM_ENABLE_GPU=ON"

Then, invoke pip with kEDM's GitHub repository as an argument:

.. code-block:: bash

    $ pip3 install git+https://github.com/keichi/kEDM.git

To install a specific version, add ``@<tag>`` to the repository URL. For
example, the following installs the v0.6.0 release of kEDM:

.. code-block:: bash

    $ pip3 install git+https://github.com/keichi/kEDM.git@v0.6.0

Install using CMake
###################

To have full control over the build, you can manually build kEDM without using
pip. This is normally needed by contributors only. If you just want to use the
Python bindings, use the pip-based method instead.

Requirements
------------

Following dependencies are required when building kEDM from source.

- `CMake <https://cmake.org/>`_ >= 3.22
- A C++ compiler supporting C++20 (currently tested on GCC and Clang)
- LAPACK (on CPU)
- CUDA Toolkit (on GPU, currently tested on CUDA >= 12)
- `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ (optional)
- MPI (optional)

On Ubuntu, install the following packages via apt:

.. code-block:: bash

    $ sudo apt-get install cmake libopenblas-serial-dev liblapacke-dev libhdf5-dev

On macOS, install the following packages via Homebrew:

.. code-block:: bash

    $ brew install cmake hdf5 libomp

Installation
------------

First clone kEDM from GitHub using git.

.. code-block:: bash

    $ git clone https://github.com/keichi/kEDM.git

Then configure and build using CMake.

.. code-block:: bash

    $ cd kEDM
    $ cmake -B build -S .
    $ cmake --build build

Below are CMake variables to customize the build:

============================= ============================================== ========
CMake flags                    Effect                                         Default
----------------------------- ---------------------------------------------- --------
``-DKEDM_ENABLE_CPU``          Enable CPU backend                              ``ON``
``-DKEDM_ENABLE_GPU``          Enable GPU backend                              ``OFF``
``-DKEDM_ENABLE_EXECUTABLES``  Build executables (e.g. ``edm-xmap``)           ``OFF``
``-DKEDM_ENABLE_PYTHON``       Build Python bindings                           ``OFF``
``-DKEDM_ENABLE_MPI``          Build MPI executables (e.g. ``edm-xmap-mpi``)   ``OFF``
``-DKEDM_ENABLE_TESTS``        Build unit tests                                ``ON``
``-DKEDM_ENABLE_LIKWID``       Enable LIKWID performance counters              ``OFF``
============================= ============================================== ========

Tips
----

- If ``-DKEDM_ENABLE_GPU=ON``, Kokkos tries to automatically detect the compute
  capability of your GPU. If this fails, you can add ``-DKokkos_ARCH_<arch>=ON`` 
  to manually set the target compute capability. For example, here are the
  flags for some recent GPUs:

  - NVIDIA H100: ``-DKokkos_ARCH_HOPPER90=ON``
  - NVIDIA A100: ``-DKokkos_ARCH_AMPERE80=ON``
  - NVIDIA V100: ``-DKokkos_ARCH_VOLTA70=ON``
  - GeForce RTX 4090: ``-DKokkos_ARCH_ADA89=ON``
  - GeForce RTX 3090: ``-DKokkos_ARCH_AMPERE86=ON``
  - GeForce RTX 2080: ``-DKokkos_ARCH_TURING75=ON``

- Similarly, Kokkos can target a specific CPU architecture. See `here <https://github.com/kokkos/kokkos/blob/master/cmake/kokkos_arch.cmake>`_ for details.

  - Intel Ice Lake SP: ``-DKokkos_ARCH_ICX``
  - AMD Zen2: ``-DKokkos_ARCH_ZEN2``
  - AMD Zen3: ``-DKokkos_ARCH_ZEN3``

- Kokkos provides a large number of CMake options to control the backends and
  features to enable, which are detailed in its
  `documentation <https://github.com/kokkos/kokkos/blob/master/BUILD.md>`_.
  Below are minimal examples for configuring Kokkos.


Testing
-------

To run the C++ unit tests, use ctest in the build directory:

.. code-block:: bash

    $ ctest

To run the Python unit tests, use pytest:

.. code-block:: bash

    $ pytest python/tests
