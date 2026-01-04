#ifndef __CCM_HPP__
#define __CCM_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

void full_sort(SimplexLUT lut, int n_lib, int n_pred, int n_partial, int Tp);

void partial_sort(SimplexLUT lut, int k, int n_lib, int n_pred, int n_partial,
                  int Tp);

void partial_sort_bitonic(SimplexLUT lut, int k, int n_lib, int n_pred,
                          int n_partial, int Tp);

std::vector<float> ccm(TimeSeries library, TimeSeries target,
                       const std::vector<int> &lib_sizes, int samples, int E,
                       int tau, int Tp, int seed, float accuracy = 1.0f);

} // namespace edm

#endif
