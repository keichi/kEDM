#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>

#include <random>

#include "ccm.hpp"
#include "knn.hpp"
#include "simplex.hpp"
#include "stats.hpp"
#include "types.hpp"

namespace edm
{

std::vector<float> ccm(TimeSeries lib, TimeSeries target,
                       const std::vector<int> &lib_sizes, int sample, int E,
                       int tau, int Tp, int seed)
{
    Kokkos::Profiling::pushRegion("EDM::ccm");

    int n_partial = (E - 1) * tau;
    int n_lib = lib.extent(0) - n_partial - Tp;
    int n_target = target.extent(0);
    int n_prediction = n_target - n_partial;

    std::vector<float> rhos;

    TmpDistances tmp("tmp_distances", n_lib, n_lib);
    SimplexLUT full_lut(n_lib, n_lib);

    calc_distances(lib, lib, tmp, n_lib, n_lib, E, tau);

    Kokkos::deep_copy(full_lut.distances, tmp);

    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_lib, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_lib), [=](int j) {
                    full_lut.distances(i, j) = sqrt(full_lut.distances(i, j));
                    full_lut.indices(i, j) = j + n_partial + Tp;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(full_lut.distances, i, Kokkos::ALL),
                Kokkos::subview(full_lut.indices, i, Kokkos::ALL));
        });

    if (seed == 0) {
        std::random_device seed_gen;
        seed = seed_gen();
    }

    std::mt19937 engine(seed);

    Kokkos::Bitset<HostSpace> mask_mirror(n_lib);
    Kokkos::Bitset<DevSpace> mask(n_lib);

    MutableTimeSeries prediction("prediction", n_prediction);

    for (int lib_size : lib_sizes) {
        SimplexLUT lut(n_prediction, E + 1);

        float rho_sum = 0.0f;

        for (int trial = 0; trial < sample; trial++) {
            mask_mirror.clear();

            for (int i = n_lib - lib_size; i < n_lib; i++) {
                int r = std::uniform_int_distribution<>(0, i)(engine);
                mask_mirror.set(mask_mirror.test(r) ? i : r);
            }

            Kokkos::deep_copy(mask, mask_mirror);

            Kokkos::parallel_for(
                "EDM::ccm::sample", n_prediction, KOKKOS_LAMBDA(int i) {
                    int selected = 0;

                    for (int j = 0; j < n_lib && selected < E + 1; j++) {
                        int idx = full_lut.indices(i, j);

                        if (mask.test(idx)) {
                            float dist = full_lut.distances(i, j);

                            lut.distances(i, selected) = dist;
                            lut.indices(i, selected) = idx;
                            selected++;
                        }
                    }
                });

            normalize_lut(lut);
            lookup(prediction, target, lut);

            float rho = edm::corrcoef(
                Kokkos::subview(target, std::make_pair(n_partial + Tp,
                                                       target.extent_int(0))),
                prediction);

            rho_sum += rho;
        }

        rhos.push_back(rho_sum / sample);
    }

    Kokkos::Profiling::popRegion();

    return rhos;
}

} // namespace edm
