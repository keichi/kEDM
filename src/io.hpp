#ifndef __IO_HPP__
#define __IO_HPP__

#include "types.hpp"

namespace edm
{

Dataset load_csv(const std::string &path);
Dataset load_hdf5(const std::string &path, const std::string &ds_name);

} // namespace edm

#endif
