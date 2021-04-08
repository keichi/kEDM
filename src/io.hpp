#ifndef __IO_HPP__
#define __IO_HPP__

#include "types.hpp"

namespace HighFive
{
class DataSet;
}

namespace edm
{

Dataset load_csv(const std::string &path);
Dataset load_hdf5(const HighFive::DataSet &dataset);

} // namespace edm

#endif
