#ifndef __SMAP_HPP__
#define __SMAP_HPP__

#include "types.hpp"

namespace edm
{

void smap(MutableTimeSeries prediction, TimeSeries library, TimeSeries target);

} // namespace edm

#endif
