#ifndef __KNN_HPP__
#define __KNN_HPP__

#include "types.hpp"

namespace edm
{

const float MIN_WEIGHT = 1e-6f;

void calc_distances(TimeSeries lib, TimeSeries pred, TmpDistances distances,
                    int n_lib, int n_pred, int E, int tau);

void calc_distances(Dataset lib, Dataset pred, TmpDistances distances,
                    int n_lib, int n_pred, int E, int tau);

void partial_sort(TmpDistances distances, SimplexLUT out, int n_lib, int n_pred,
                  int top_k, int shift);

void knn(TimeSeries lib, TimeSeries pred, SimplexLUT out, TmpDistances tmp,
         int E, int tau, int Tp, int top_k);

void knn(Dataset lib, Dataset pred, SimplexLUT out, TmpDistances tmp, int E,
         int tau, int Tp, int top_k);

void normalize_lut(SimplexLUT lut);

} // namespace edm

#endif
