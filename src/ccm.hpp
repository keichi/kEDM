#ifndef __CCM_HPP__
#define __CCM_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

void full_sort(TmpDistances distances, TmpIndices indices, int n_lib,
               int n_pred, int n_partial, int Tp);

void full_sort_with_scratch(TmpDistances distances, TmpIndices indices,
                            int n_lib, int n_pred, int n_partial, int Tp);

void full_sort_cpu(TmpDistances distances, TmpIndices indices, int n_lib,
                   int n_pred, int n_partial, int Tp);

void full_sort_radix(TmpDistances distances, TmpIndices indices, int n_lib,
                     int n_pred, int n_partial, int Tp);

void partial_sort(TmpDistances distances, TmpIndices indices, int k, int n_lib,
                  int n_pred, int n_partial, int Tp);

void partial_sort_cpu(TmpDistances distances, TmpIndices indices, int k,
                      int n_lib, int n_pred, int n_partial, int Tp);

std::vector<float> ccm(TimeSeries library, TimeSeries target,
                       const std::vector<int> &lib_sizes, int samples, int E,
                       int tau, int Tp, int seed, float accuracy = 1.0f);

std::vector<float> ccm_naive(TimeSeries lib, TimeSeries target,
                             const std::vector<int> &lib_sizes, int sample,
                             int E, int tau, int Tp, int seed);

} // namespace edm

#endif
