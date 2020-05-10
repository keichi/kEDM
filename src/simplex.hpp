#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include "types.hpp"

namespace edm
{

void simplex(TimeSeries prediction, const TimeSeries target, const LUT &lut);

} // namespace edm

#endif
