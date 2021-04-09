#include <iostream>

// #include <mkl_lapacke.h>
#include <cusolverDn.h>

#include "knn.hpp"
#include "smap.hpp"
#include "types.hpp"

#define CUDA_CHECK(CALL)                                                       \
    do {                                                                       \
        cudaError_t error = CALL;                                              \
        assert(error == cudaSuccess);                                          \
    } while (0)
#define CUSOLVER_CHECK(CALL)                                                   \
    do {                                                                       \
        cusolverStatus_t status = CALL;                                        \
        assert(status == CUSOLVER_STATUS_SUCCESS);                             \
    } while (0)

namespace edm
{

void smap(MutableTimeSeries prediction, TimeSeries library, TimeSeries target)
{
    const float theta = 0.0f;

    const int E = 20, tau = 1, Tp = 1;
    const int shift = (E - 1) * tau + Tp;
    const int n_library = library.size() - shift;
    const int n_target = target.size() - shift + Tp;

    TmpDistances distances("tmp_distances", n_target, n_library);

    calc_distances(library, target, distances, n_library, n_target, E, tau);

    Kokkos::View<float **, DevSpace> A("A", n_library, E + 1);
    Kokkos::View<float *, DevSpace> b("b", n_library), X("X", E + 1),
        w("w", n_library);

    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t work_bytes = 0;
    CUSOLVER_CHECK(cusolverDnSSgels_bufferSize(
        handle, n_library, E + 1, 1, nullptr, n_library, nullptr, n_library,
        nullptr, E + 1, nullptr, &work_bytes));

    float *work = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&work, work_bytes));

    int *dinfo = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dinfo, sizeof(int)));

    for (int i = 0; i < n_target; i++) {
        float d = 0.0f;

        Kokkos::parallel_reduce(
            "EDM::smap::sum", n_library,
            KOKKOS_LAMBDA(const int j, float &sum) { sum += distances(i, j); },
            d);

        d /= n_library;

        Kokkos::parallel_for(
            "EDM::smap::weights", n_library, KOKKOS_LAMBDA(const int j) {
                w(j) = exp(-theta * distances(i, j) / d);
            });

        Kokkos::parallel_for(
            "EDM::smap::prepare", n_library, KOKKOS_LAMBDA(const int j) {
                A(j, 0) = w(j);
                for (int k = 0; k < E; k++) {
                    A(j, k + 1) = w(j) * library(j + k * tau);
                }
                b(j) = w(j) * library(j + shift);
            });

        int niter = 0;
        CUSOLVER_CHECK(cusolverDnSSgels(
            handle, n_library, E + 1, 1, A.data(), n_library, b.data(),
            n_library, X.data(), E + 1, work, work_bytes, &niter, dinfo));

        // LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', n_library, E + 1, 1, &A(0, 0),
                      // E + 1, &b(0), 1);
    }

    CUDA_CHECK(cudaFree(work));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}

} // namespace edm
