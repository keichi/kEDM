#ifndef __EDIM_HPP__
#define __EDIM_HPP__

#include "types.hpp"

namespace edm
{

uint32_t edim(const TimeSeries &ts, int E_max, int tau, int Tp);

} // namespace edm

#endif
