#ifndef __STATS_HPP__
#define __STATS_HPP__

#include "types.hpp"

namespace edm
{

// Simple sum-of-products state for corrcoef (faster parallel reduction)
typedef struct corrcoef_simple_state {
    float sum_x, sum_y, sum_x2, sum_y2, sum_xy;
    int n;

    KOKKOS_INLINE_FUNCTION corrcoef_simple_state()
        : sum_x(0.0f), sum_y(0.0f), sum_x2(0.0f), sum_y2(0.0f), sum_xy(0.0f),
          n(0)
    {
    }

    KOKKOS_INLINE_FUNCTION corrcoef_simple_state(float x, float y)
        : sum_x(x), sum_y(y), sum_x2(x * x), sum_y2(y * y), sum_xy(x * y), n(1)
    {
    }

    KOKKOS_INLINE_FUNCTION
    float rho() const
    {
        const float mean_x = sum_x / n;
        const float mean_y = sum_y / n;
        const float var_x = sum_x2 / n - mean_x * mean_x;
        const float var_y = sum_y2 / n - mean_y * mean_y;
        const float cov_xy = sum_xy / n - mean_x * mean_y;
        return cov_xy / sqrt(var_x * var_y);
    }

    KOKKOS_INLINE_FUNCTION
    corrcoef_simple_state &operator+=(const corrcoef_simple_state &src)
    {
        sum_x += src.sum_x;
        sum_y += src.sum_y;
        sum_x2 += src.sum_x2;
        sum_y2 += src.sum_y2;
        sum_xy += src.sum_xy;
        n += src.n;
        return *this;
    }

} CorrcoefSimpleState;

// Welford-based state for corrcoef (better numerical stability)
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
} CorrcoefState;

float corrcoef(TimeSeries x, TimeSeries y);
void corrcoef(CrossMap rho, Dataset ds, TimeSeries x);
float mae(TimeSeries x, TimeSeries y);
float mse(TimeSeries x, TimeSeries y);

} // namespace edm

namespace Kokkos
{

template <> struct reduction_identity<edm::CorrcoefSimpleState> {
    KOKKOS_FORCEINLINE_FUNCTION static edm::CorrcoefSimpleState sum()
    {
        return edm::CorrcoefSimpleState();
    }
};

template <> struct reduction_identity<edm::CorrcoefState> {
    KOKKOS_FORCEINLINE_FUNCTION static edm::CorrcoefState sum()
    {
        return edm::CorrcoefState();
    }
};

} // namespace Kokkos

#endif
