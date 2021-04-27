#ifndef __STATS_HPP__
#define __STATS_HPP__

#include "types.hpp"

namespace edm
{
typedef struct corrcoef_state {
    int n;
    float x_mean, y_mean, x_m2, y_m2, xy_m2;

    KOKKOS_INLINE_FUNCTION corrcoef_state()
        : n(0), x_mean(0.0f), y_mean(0.0f), x_m2(0.0f), y_m2(0.0f), xy_m2(0.0f)
    {
    }

    KOKKOS_INLINE_FUNCTION corrcoef_state(float x, float y)
        : n(1), x_mean(x), y_mean(y), x_m2(0.0f), y_m2(0.0f), xy_m2(0.0f)
    {
    }

    KOKKOS_INLINE_FUNCTION
    float rho() const { return xy_m2 / sqrt(x_m2 * y_m2); }

    KOKKOS_INLINE_FUNCTION
    corrcoef_state &operator+=(const corrcoef_state &src)
    {
        const int n_a = n, n_b = src.n;
        const int n_ab = n_a + n_b;

        const float x_delta = src.x_mean - x_mean;
        const float y_delta = src.y_mean - y_mean;

        n = n_ab;

        if (n == 0) return *this;

        x_mean += x_delta * n_b / n_ab;
        y_mean += y_delta * n_b / n_ab;

        x_m2 += src.x_m2 + x_delta * x_delta * n_a * n_b / n_ab;
        y_m2 += src.y_m2 + y_delta * y_delta * n_a * n_b / n_ab;
        xy_m2 += src.xy_m2 + x_delta * y_delta * n_a * n_b / n_ab;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION void
    operator+=(const volatile corrcoef_state &src) volatile
    {
        const int n_a = n, n_b = src.n;
        const int n_ab = n_a + n_b;

        const float x_delta = src.x_mean - x_mean;
        const float y_delta = src.y_mean - y_mean;

        n = n_ab;

        if (n == 0) return;

        x_mean += x_delta * n_b / n_ab;
        y_mean += y_delta * n_b / n_ab;

        x_m2 += src.x_m2 + x_delta * x_delta * n_a * n_b / n_ab;
        y_m2 += src.y_m2 + y_delta * y_delta * n_a * n_b / n_ab;
        xy_m2 += src.xy_m2 + x_delta * y_delta * n_a * n_b / n_ab;
    }
} CorrcoefState;

float corrcoef(TimeSeries x, TimeSeries y);
void corrcoef(CrossMap rho, Dataset ds, TimeSeries x);
float mae(TimeSeries x, TimeSeries y);
float mse(TimeSeries x, TimeSeries y);

} // namespace edm

namespace Kokkos
{

template <> struct reduction_identity<edm::CorrcoefState> {
    KOKKOS_FORCEINLINE_FUNCTION static edm::CorrcoefState sum()
    {
        return edm::CorrcoefState();
    }
};

} // namespace Kokkos

#endif
