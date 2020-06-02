#ifndef __KNN_HPP__
#define __KNN_HPP__

#include "types.hpp"

namespace edm
{

const float MIN_WEIGHT = 1e-6f;

void knn(const TimeSeries &library, const TimeSeries &target, LUT &out,
         LUT &tmp, uint32_t E, int32_t tau, int32_t Tp, uint32_t top_k);

void normalize_lut(LUT &lut);

} // namespace edm

#endif
