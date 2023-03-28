#ifndef __SMAP_HPP__
#define __SMAP_HPP__

#include "types.hpp"

namespace edm
{

void smap(MutableTimeSeries result, TimeSeries lib, TimeSeries pred,
          TimeSeries target, int E, int tau, int Tp, float theta);

} // namespace edm

#endif
