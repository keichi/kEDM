#ifndef __XMAP_HPP__
#define __XMAP_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

void xmap(CrossMap &result, const Dataset &ds, const TimeSeries &library,
          const std::vector<int> &edims, int E_max, int tau, int Tp);

} // namespace edm

#endif
