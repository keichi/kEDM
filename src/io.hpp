#ifndef __IO_HPP__
#define __IO_HPP__

#include "types.hpp"

#ifdef HAVE_HDF5
namespace HighFive
{
class DataSet;
}
#endif

namespace edm
{

Dataset load_csv(const std::string &path);
#ifdef HAVE_HDF5
Dataset load_hdf5(const HighFive::DataSet &dataset);
#endif

} // namespace edm

#endif
