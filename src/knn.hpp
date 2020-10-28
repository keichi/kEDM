#ifndef __KNN_HPP__
#define __KNN_HPP__

#include "types.hpp"

namespace edm
{

const float MIN_WEIGHT = 1e-6f;

void calc_distances(TimeSeries library, TimeSeries target,
                    TmpDistances distances, int n_library, int n_target, int E,
                    int tau);

void partial_sort(TmpDistances distances, LUT out, int n_library, int n_target,
                  int top_k, int shift);

void knn(TimeSeries library, TimeSeries target, LUT out, TmpDistances tmp,
         int E, int tau, int Tp, int top_k);

void normalize_lut(LUT lut);

} // namespace edm

#endif
