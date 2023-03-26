#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>

#include "types.hpp"

namespace edm
{

std::ostream &operator<<(std::ostream &os, const SimplexLUT &lut);

}

#endif
