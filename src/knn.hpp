#ifndef __KNN_HPP__
#define __KNN_HPP__

#include "types.hpp"

namespace edm
{

const auto MIN_WEIGHT = 1e-6f;

class NearestNeighbors
{
    LUT _cache;

public:
    NearestNeighbors(LUT cache) : _cache(cache) {}

    void run(const TimeSeries &library, const TimeSeries &target, LUT &lut,
             int E, int tau, int Tp, int top_k);
};

void normalize_lut(LUT &lut);

} // namespace edm

#endif
