Installation
############

Installing via pip
------------------

Requirements
============

- Python >= 3.6
- pip >= 19.3

.. code-block:: bash

    $ pip install kedm

Installing from source
----------------------

Requirements
============

When building kEDM from source, following tools and libraries are required:

- `CMake <https://cmake.org/>`_ >= 3.16
- LAPACK (on CPU) or cuBLAS (on GPU)
- `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ (optional)
- MPI (optional)

On Ubuntu, install the following packages via apt:

- CMake: ``cmake``
- LAPACK: ``libopenblas-dev`` and ``liblapacke-dev``
- HDF5: ``libhdf5-dev`` or ``libhdf5-openmpi-dev``
- MPI: ``libopenmpi-dev``


Building
=============

Clone kEDM from GitHub using git. Note that the ``--recursive`` flag is
required because third-party libraries are bundled as submodules.

.. code-block:: bash

    $ git clone --recursive https://github.com/keichi/kEDM.git
    $ cd kEDM
    $ mkdir build
    $ cd build

Then configure and build using CMake.

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
====

- If ``-DKEDM_ENABLE_GPU=ON``, Kokkos tries to automatically detect the compute
  capability of your GPU. If this fails, you can add ``-DKokkos_ARCH_<arch>`` to
  manually set the target compute capability. For example, here are the flags
  for latest GPUs:

  - NVIDIA A100: ``-DKokkos_ARCH_AMPERE80``
  - NVIDIA V100: ``-DKokkos_ARCH_VOLTA70``
  - GeForce RTX 3090: ``-DKokkos_ARCH_AMPERE86``
  - GeForce RTX 2080: ``-DKokkos_ARCH_TURING75``

- Similarly, see `here <https://github.com/kokkos/kokkos/blob/master/cmake/kokkos_arch.cmake>`_ for details.

  - Intel Sky Lake: ``-DKokkos_ARCH_SKX``
  - AMD Zen2: ``-DKokkos_ARCH_ZEN2``

- Kokkos provides a large number of CMake options to control the backends and
  features to enable, which are detailed in its
  `documentation <https://github.com/kokkos/kokkos/blob/master/BUILD.md>`_.
  Below are minimal examples for configuring Kokkos.


Testing kEDM
============

Run ``ctest`` within your build directory to execute the unit tests.

.. code-block:: bash

    $ ctest
