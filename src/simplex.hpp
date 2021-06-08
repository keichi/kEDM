#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include "types.hpp"

namespace edm
{

void simplex(MutableTimeSeries prediction, TimeSeries library,
             TimeSeries target, int E, int tau, int Tp);

void simplex(MutableTimeSeries prediction, TimeSeries target, SimplexLUT lut);

} // namespace edm

#endif
