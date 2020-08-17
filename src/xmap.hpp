#ifndef __XMAP_HPP__
#define __XMAP_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

void group_ts(std::vector<Targets> &groups, const std::vector<int> &edims,
              int E_max);

void xmap(CrossMap &result, const Dataset &ds, const TimeSeries &library,
          const std::vector<Targets> &groups, std::vector<LUT> &luts,
          LUT &tmp_lut, int E_max, int tau, int Tp);

} // namespace edm

#endif
