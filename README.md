# kEDM

[![build](https://github.com/keichi/kEDM/workflows/build/badge.svg)](https://github.com/keichi/kEDM/actions?query=workflow%3Abuild) [![Documentation Status](https://readthedocs.org/projects/kedm/badge/?version=latest)](https://kedm.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/kedm.svg)](https://badge.fury.io/py/kedm)

kEDM (Kokkos-EDM) is a high-performance implementation of the [Empirical
Dynamical Modeling (EDM)](https://sugiharalab.github.io/EDM_Documentation/)
framework. The goal of kEDM is to provide an optimized and parallelized
implementation of EDM algorithms for HPC hardware (Intel Xeon, AMD EPYC,
NVIDIA GPUs, Fujitsu A64FX, etc.) while ensuring compatibility with the
reference implementation ([cppEDM](https://github.com/SugiharaLab/cppEDM))

Following EDM algorithms are currently implemented in kEDM:

- Simplex projection [1]
- Sequential Locally Weighted Global Linear Maps (S-Map) [2]
- Convergent Cross Mapping (CCM) [3]

## Citing

Please cite the following paper if you find kEDM useful:

Keichi Takahashi, Wassapon Watanakeesuntorn, Kohei Ichikawa, Joseph Park,
Ryousei Takano, Jason Haga, George Sugihara, Gerald M. Pao, "kEDM: A
Performance-portable Implementation of Empirical Dynamical Modeling," Practice
& Experience in Advanced Research Computing (PEARC 2021), Jul. 2021.

## References

1. George Sugihara, Robert May, "Nonlinear forecasting as a way of
   distinguishing chaos from measurement error in time series," Nature, vol.
   344, pp. 734-741,  1990. [10.1038/344734a0](https://doi.org/10.1038/344734a0)
2. George Sugihara, "Nonlinear forecasting for the classification of natural
   time series. Philosophical Transactions," Physical Sciences and Engineering,
   vol. 348, no. 1688, pp. 477–495, 1994.
   [10.1098/rsta.1994.0106](https://doi.org/10.1098/rsta.1994.0106)
3. George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael
   Fogarty, Stephan Munch, "Detecting Causality in Complex Ecosystems,"
   Science, vol. 338, pp. 496-500, 2012.
   [10.1126/science.1227079](https://doi.org/10.1126/science.1227079)
