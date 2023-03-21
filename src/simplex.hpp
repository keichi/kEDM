#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include "types.hpp"

namespace edm
{

// Univariate
void simplex(MutableTimeSeries prediction, TimeSeries library,
             TimeSeries target, int E, int tau, int Tp);

void simplex(MutableTimeSeries prediction, TimeSeries target, SimplexLUT lut);

// Multivariate
void simplex(MutableDataset prediction, Dataset library, Dataset target, int E,
             int tau, int Tp);

void simplex(MutableDataset prediction, Dataset target, SimplexLUT lut);

} // namespace edm

#endif
