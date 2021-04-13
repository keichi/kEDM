#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cublas_v2.h>
#else
#include <mkl_lapacke.h>
#endif

#include "smap.hpp"
#include "types.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#define CUDA_CHECK(CALL)                                                       \
    do {                                                                       \
        cudaError_t error = CALL;                                              \
        assert(error == cudaSuccess);                                          \
    } while (0);
#define CUBLAS_CHECK(CALL)                                                     \
    do {                                                                       \
        cublasStatus_t status = CALL;                                          \
        assert(status == CUBLAS_STATUS_SUCCESS);                               \
    } while (0);
#else
#define LAPACK_CHECK(CALL)                                                     \
    do {                                                                       \
        int status = CALL;                                                     \
        assert(status == 0);                                                   \
    } while (0);
#endif

namespace edm
{

void smap(MutableTimeSeries prediction, TimeSeries library, TimeSeries target,
          int E, int tau, int Tp, float theta)
{
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

#ifdef KOKKOS_ENABLE_CUDA
    const int batch_size =
        std::max(1ul, (1 << 30) / (n_library * (E + 1) * sizeof(float)));
#else
    const int batch_size = 1;
#endif

    Kokkos::View<float **, DevSpace> d("distances", n_library, batch_size);
    Kokkos::View<float ***, DevSpace> A("design", n_library, E + 1, batch_size);
    Kokkos::View<float **, DevSpace> b("response", n_library, batch_size);

#ifdef KOKKOS_ENABLE_CUDA
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
#endif

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
                            float diff = library(j + k * tau) -
                                         target(offset + i + k * tau);
                            dist += diff * diff;
                        }

                        if (library.data() + j == target.data() + offset + i) {
                            dist = FLT_MAX;
                        } else {
                            dist = sqrt(dist);
                        }

                        d(j, i) = dist;
                    });

                float d_mean = 0.0f;

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, n_library),
                    [=](int j, float &sum) {
                        sum += d(j, i) < FLT_MAX ? d(j, i) : 0.0f;
                    },
                    d_mean);

                d_mean /= n_library;

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

#ifdef KOKKOS_ENABLE_CUDA
        CUBLAS_CHECK(cublasSgelsBatched(handle, CUBLAS_OP_N, n_library, E + 1,
                                        1, As, n_library, bs, n_library, &info,
                                        dev_infos, this_batch_size));
        assert(info == 0);
#else
        LAPACK_CHECK(LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', n_library, E + 1, 1,
                                   A.data(), E + 1, b.data(), 1));
#endif

        Kokkos::parallel_for(
            "EDM::smap::postprocess", this_batch_size, KOKKOS_LAMBDA(int i) {
                float pred = b(0, i);

                for (int k = 0; k < E; k++) {
                    pred += b(k + 1, i) * target(offset + i + k * tau);
                }

                prediction(i + offset) = pred;
            });
    }

#ifdef KOKKOS_ENABLE_CUDA
    CUBLAS_CHECK(cublasDestroy(handle));

    Kokkos::kokkos_free(As);
    Kokkos::kokkos_free(bs);
    Kokkos::kokkos_free(dev_infos);
#endif
}

} // namespace edm
