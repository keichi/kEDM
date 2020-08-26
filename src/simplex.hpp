#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include "types.hpp"

namespace edm
{

void simplex(MutableTimeSeries prediction, TimeSeries target, LUT lut);

} // namespace edm

#endif
