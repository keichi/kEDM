#ifndef __XMAP_HPP__
#define __XMAP_HPP__

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace edm
{

void xmap(const Dataset &ds, const std::vector<int> &edims, CrossMap &rho,
          int E_max);

} // namespace edm

#endif
