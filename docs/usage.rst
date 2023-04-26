Utilities
#################

Pairwise Cross Mapping
==============================

Overview
--------

``edm-xmap`` performs pairwise cross mapping between many time series using
Simplex projection. The input is given as a 2D array where a column
corresponds to a time series. The output is a 2D array where each element
represents the cross map rho between two time series. Optionally, pairwise
cross-correlation can also be computed.

Both the input and output are stored as HDF5 files. The output HDF5 file 
contains up to four datasets, which are:

- ``ccm``: Cross map rho
- ``e``: Optimal embedding dimension
- ``rho``: Pairwise cross-correlation (if ``--rho`` is enabled)
- ``rhodiff``: ``ccm - abs(rho)`` (if ``--rho-diff`` is enabled)

Command-line options
--------------------

.. code-block:: bash

    edm-xmap: All-to-all Convergent Cross Mapping Tool

    Usage:
      edm-xmap [OPTION...] input output
      -d, --dataset arg           HDF5 dataset name (default: "values")
      -e, --max-embedding-dim arg Embedding dimension (default: 20)
      -t, --tau arg               Time delay (default: 1)
      --rho                       Compute cross correlation (default: false)
      --rho-diff                  Compute rho diff(default: false)
      -v, --verbose               Enable verbose output
      -h, --help                  Show this help

Example
-------

Here, an HDF5 file ``Fish1_150a_Normo.h5`` containing 154 time series with 1,600
time steps, is processed using ``edm-xmap``.

.. code-block:: bash

    $ h5ls Fish1_150a_Normo.h5
    names                    Dataset {154}
    values                   Dataset {1600, 154}

.. code-block:: bash

    $ ./edm-xmap --rho --rho-diff Fish1_150a_Normo.h5 Fish1_150a_Normo_xmap.h5

.. code-block:: bash

    $ h5ls Fish1_150a_Normo_xmap.h5
    ccm                      Dataset {154, 154}
    e                        Dataset {154}
    rho                      Dataset {154, 154}
    rhodiff                  Dataset {154, 154}

Micro benchmarks
================

Overview
--------

kEDM includes several micro benchmarks to quickly measure the performance of
bottleneck kernels using dummy datasets.

Command-line options
--------------------

.. code-block:: bash

    knn-bench: k-Nearest Neighbors Search Benchmark

    Usage:
      knn-bench [OPTION...]
      -l, --length arg        Length of time series (default: 10,000)
      -e, --embedding-dim arg Embedding dimension (default: 20)
      -t, --tau arg           Time delay (default: 1)
      -i, --iteration arg     Number of iterations (default: 10)
      -v, --verbose           Enable verbose logging (default: false)
      -h, --help              Show this help

.. code-block:: bash

    lookup-bench: kNN Lookup Benchmark

    Usage:
      build-cuda/lookup-bench [OPTION...]
      -l, --length arg        Length of time series (default: 10,000)
      -n, --num-ts arg        Number of time series (default: 10,000)
      -e, --embedding-dim arg Embedding dimension (default: 20)
      -t, --tau arg           Time delay (default: 1)
      -i, --iteration arg     Number of iterations (default: 10)
      -v, --verbose           Enable verbose logging (default: false)
      -h, --help              Show this help
