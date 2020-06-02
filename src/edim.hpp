#ifndef __EDIM_HPP__
#define __EDIM_HPP__

#include "types.hpp"

namespace edm
{

uint32_t edim(const TimeSeries &ts, uint32_t E_max, int32_t tau, int32_t Tp);

} // namespace edm

#endif
