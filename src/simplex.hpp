#ifndef __SIMPLEX_HPP__
#define __SIMPLEX_HPP__

#include <Kokkos_Core.hpp>

namespace edm {

void lookup(Kokkos::View<float **> ds, Kokkos::View<unsigned int *> targets,
            Kokkos::View<float **> distances,
            Kokkos::View<unsigned int **> indices, const int E)
{
    Kokkos::Timer timer;

    Kokkos::parallel_for(
        "lookup", Kokkos::TeamPolicy<>(targets.extent(0), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, distances.extent(0)), [=](size_t j) {
                    float pred = 0.0f;

                    for (int e = 0; e < E; e++) {
                        pred += ds(indices(j, e), targets(i)) * indices(j, e);
                    }
                });
        });

    std::cout << "elapsed: " << timer.seconds() << " s" << std::endl;
}

}

#endif
