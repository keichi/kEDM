#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cublas_v2.h>
#endif

#include "smap.hpp"
#include "types.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#define CUDA_CHECK(CALL)                                                       \
    do {                                                                       \
        cudaError_t error = CALL;                                              \
        if (error != cudaSuccess) {                                            \
            throw std::runtime_error("CUDA error: " +                          \
                                     cudaGetErrorString(error));               \
        };                                                                     \
    } while (0);
#define CUBLAS_CHECK(CALL)                                                     \
    do {                                                                       \
        cublasStatus_t status = CALL;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error("cuBLAS error: " +                        \
                                     std::to_string(status));                  \
        }                                                                      \
    } while (0);
#endif

extern "C" {
void sgels_(char const *trans, int const *m, int const *n, int const *nrhs,
            float *A, int const *lda, float *B, int const *ldb, float *work,
            int const *lwork, int *info);
}

namespace edm
{

void smap(MutableTimeSeries result, TimeSeries lib, TimeSeries pred,
          TimeSeries target, int E, int tau, int Tp, float theta)
{
    const int n_partial = (E - 1) * tau;
    const int n_lib = lib.extent(0) - n_partial - Tp;
    const int n_pred = pred.extent(0) - n_partial;

    if (E <= 0) {
        throw std::invalid_argument("E must be greater than zero");
    } else if (tau <= 0) {
        throw std::invalid_argument("tau must be greater than zero");
    } else if (Tp < 0) {
        throw std::invalid_argument("Tp must be greater or equal to zero");
    } else if (n_lib <= 0) {
        throw std::invalid_argument("lib size is too small");
    } else if (n_pred <= 0) {
        throw std::invalid_argument("pred size is too small");
    } else if (pred.extent(0) != target.extent(0)) {
        throw std::invalid_argument("pred size and target size must be equal");
    }

#ifdef KOKKOS_ENABLE_CUDA
    // Make sure the design matrices fit within 4GiB
    const int batch_size =
        std::max(1ul, (1ul << 32) / (n_lib * (E + 1) * sizeof(float)));
#else
    // For now we do not use batched kernels on CPU
    const int batch_size = 1;
#endif

    Kokkos::View<float **, Kokkos::LayoutLeft, DevSpace> d("distances", n_lib,
                                                           batch_size);
    Kokkos::View<float ***, Kokkos::LayoutLeft, DevSpace> A("design", n_lib,
                                                            E + 1, batch_size);
    Kokkos::View<float **, Kokkos::LayoutLeft, DevSpace> b("response", n_lib,
                                                           batch_size);

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
#else
    int m = n_lib, n = E + 1, nrhs = 1, lda = n_lib, ldb = n_lib;
    int work_size = 0, lwork = -1, info = 0;

    sgels_("N", &m, &n, &nrhs, A.data(), &lda, b.data(), &ldb,
           reinterpret_cast<float *>(&work_size), &lwork, &info);

    float *work = (float *)Kokkos::kokkos_malloc<>(work_size * sizeof(float));
#endif

    for (int offset = 0; offset < n_pred; offset += batch_size) {
        int this_batch_size = std::min(batch_size, n_pred - offset);

        Kokkos::parallel_for(
            "EDM::smap::preprocess",
            Kokkos::TeamPolicy<>(this_batch_size, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
                int i = member.league_rank();

                // Compute Euclidean distances in state space
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                        float dist = 0.0f;
                        for (int k = 0; k < E; k++) {
                            float diff =
                                lib(j + k * tau) - pred(offset + i + k * tau);
                            dist += diff * diff;
                        }

                        d(j, i) = dist == 0.0f ? FLT_MAX : sqrt(dist);
                    });

                member.team_barrier();

                float d_mean = 0.0f;

                // Compute mean distance
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, n_lib),
                    [=](int j, float &sum) {
                        sum += d(j, i) < FLT_MAX ? d(j, i) : 0.0f;
                    },
                    d_mean);

                d_mean /= n_lib;

                // Fill out design matrices and response vectors
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                        float w = exp(-theta * d(j, i) / d_mean);

                        A(j, 0, i) = w;
                        for (int k = 0; k < E; k++) {
                            A(j, k + 1, i) = w * lib(j + k * tau);
                        }
                        b(j, i) = w * target(j + n_partial + Tp);
                    });
            });

        // Invoke least-squares solver
#ifdef KOKKOS_ENABLE_CUDA
        CUBLAS_CHECK(cublasSgelsBatched(handle, CUBLAS_OP_N, n_lib, E + 1, 1,
                                        As, n_lib, bs, n_lib, &info, dev_infos,
                                        this_batch_size));

        if (info != 0) {
            throw std::runtime_error("cublasSgelsBatched returned error: " +
                                     std::to_string(info));
        }
#else
        sgels_("N", &m, &n, &nrhs, A.data(), &lda, b.data(), &ldb, work,
               &work_size, &info);

        if (info != 0) {
            throw std::runtime_error("sgels_ returned error: " +
                                     std::to_string(info));
        }
#endif

        Kokkos::parallel_for(
            "EDM::smap::postprocess", this_batch_size, KOKKOS_LAMBDA(int i) {
                float p = b(0, i);

                for (int k = 0; k < E; k++) {
                    p += b(k + 1, i) * lib(offset + i + k * tau);
                }

                result(i + offset) = p;
            });
    }

#ifdef KOKKOS_ENABLE_CUDA
    CUBLAS_CHECK(cublasDestroy(handle));

    Kokkos::kokkos_free(As);
    Kokkos::kokkos_free(bs);
    Kokkos::kokkos_free(dev_infos);
#else
    Kokkos::kokkos_free(work);
#endif
}

} // namespace edm
