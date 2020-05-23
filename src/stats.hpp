#ifndef __STATS_HPP__
#define __STATS_HPP__

#include "types.hpp"

namespace edm
{

KOKKOS_INLINE_FUNCTION float corrcoef(const TimeSeries &x, const TimeSeries &y)
{
#ifndef KOKKOS_ENABLE_CUDA
    using std::min;
    using std::sqrt;
#endif

    const auto n = min(x.size(), y.size());
    auto mean_x = 0.0f, mean_y = 0.0f;
    auto sum_xy = 0.0f, sum_x2 = 0.0f, sum_y2 = 0.0f;

    for (auto i = 0u; i < n; i++) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    for (auto i = 0u; i < n; i++) {
        auto diff_x = x[i] - mean_x;
        auto diff_y = y[i] - mean_y;

        sum_xy += diff_x * diff_y;
        sum_x2 += diff_x * diff_x;
        sum_y2 += diff_y * diff_y;
    }

    return sum_xy / sqrt(sum_x2 * sum_y2);
}

} // namespace edm

#endif
