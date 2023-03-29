#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include "types.hpp"

namespace edm
{

// Univariate
void simplex(MutableTimeSeries result, TimeSeries lib, TimeSeries pred,
             TimeSeries target, int E, int tau, int Tp);

// Multivariate
void simplex(MutableTimeSeries result, Dataset lib, Dataset pred,
             TimeSeries target, int E, int tau, int Tp);

void lookup(MutableTimeSeries result, TimeSeries target, SimplexLUT lut);

} // namespace edm

#endif
