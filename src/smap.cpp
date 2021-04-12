#include <iostream>

#include <cublas_v2.h>

#include "smap.hpp"
#include "types.hpp"

#define CUDA_CHECK(CALL)                                                       \
    do {                                                                       \
        cudaError_t error = CALL;                                              \
        assert(error == cudaSuccess);                                          \
    } while (0)
#define CUBLAS_CHECK(CALL)                                                     \
    do {                                                                       \
        cublasStatus_t status = CALL;                                          \
        assert(status == CUBLAS_STATUS_SUCCESS);                               \
    } while (0)

namespace edm
{

void smap(MutableTimeSeries prediction, TimeSeries library, TimeSeries target,
          int E, int tau, int Tp, float theta)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;
    const int batch_size = 1000;

    Kokkos::View<float **, DevSpace> d("dist", n_library, batch_size);
    Kokkos::View<float ***, DevSpace> A("A", n_library, E + 1, batch_size);
    Kokkos::View<float **, DevSpace> b("b", n_library, batch_size);

    int info;
    float **As =
        (float **)Kokkos::kokkos_malloc<>(batch_size * sizeof(float *));
    float **bs =
        (float **)Kokkos::kokkos_malloc<>(batch_size * sizeof(float *));
    int *dev_infos = (int *)Kokkos::kokkos_malloc<>(batch_size * sizeof(int));

    Kokkos::parallel_for(
        "EDM::smap::pointers", batch_size, KOKKOS_LAMBDA(int i) {
            As[i] = &A(0, 0, i);
            bs[i] = &b(0, i);
        });

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int offset = 0; offset < n_target; offset += batch_size) {
        int this_batch_size = std::min(batch_size, n_target - offset);

        Kokkos::parallel_for(
            "EDM::smap::preprocess",
            Kokkos::TeamPolicy<>(this_batch_size, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
                int i = member.league_rank();

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                        float dist = 0.0f;
                        for (int k = 0; k < E; k++) {
                            float diff = library(offset + i + k * tau) -
                                         target(j + k * tau);
                            dist += diff * diff;
                        }

                        if (library.data() + offset + i == target.data() + j) {
                            dist = FLT_MAX;
                        }

                        d(j, i) = dist;
                    });

                float d_mean = 0.0f;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, n_library),
                    [=](int j, float &sum) {
                        if (d(j, i) == FLT_MAX) return;
                        sum += d(j, i);
                    },
                    d_mean);

                Kokkos::single(Kokkos::PerTeam(member),
                               [&] { d_mean /= n_library; });

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                        float w = exp(-theta * d(j, i) / d_mean);

                        A(j, 0, i) = w;
                        for (int k = 0; k < E; k++) {
                            A(j, k + 1, i) = w * library(j + k * tau);
                        }
                        b(j, i) = w * library(j + shift);
                    });
            });

        CUBLAS_CHECK(cublasSgelsBatched(handle, CUBLAS_OP_N, n_library, E + 1,
                                        1, As, n_library, bs, n_library, &info,
                                        dev_infos, this_batch_size));

        Kokkos::parallel_for(
            "EDM::smap::postprocess", this_batch_size, KOKKOS_LAMBDA(int i) {
                float pred = b(0, i);

                for (int k = 0; k < E; k++) {
                    pred += b(k + 1, i) * library(i + k * tau);
                }

                prediction(i + offset) = pred;
            });
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    Kokkos::kokkos_free(As);
    Kokkos::kokkos_free(bs);
    Kokkos::kokkos_free(dev_infos);
}

} // namespace edm
