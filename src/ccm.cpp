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

std::vector<float> ccm(TimeSeries library, TimeSeries target,
                       const std::vector<int> &lib_sizes, int samples, int E,
                       int tau, int Tp)
{
    Kokkos::Profiling::pushRegion("EDM::ccm");

    int shift = (E - 1) * tau + Tp;
    int n_library = library.extent(0) - shift;
    int n_target = target.extent(0);
    int n_prediction = n_target - shift + Tp;

    std::vector<float> rhos;

    TmpDistances tmp("tmp_distances", n_library, n_library);
    SimplexLUT full_lut(n_library, n_library);

    calc_distances(library, library, tmp, n_library, n_library, E, tau);

    Kokkos::deep_copy(full_lut.distances, tmp);

    Kokkos::parallel_for(
        "EDM::ccm::sort", Kokkos::TeamPolicy<>(n_library, Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &member) {
            int i = member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, n_library), [=](int j) {
                    full_lut.distances(i, j) = sqrt(full_lut.distances(i, j));
                    full_lut.indices(i, j) = j + shift;
                });

            member.team_barrier();

            Kokkos::Experimental::sort_by_key_team(
                member, Kokkos::subview(full_lut.distances, i, Kokkos::ALL),
                Kokkos::subview(full_lut.indices, i, Kokkos::ALL));
        });

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    Kokkos::Bitset<HostSpace> mask_mirror(n_library);
    Kokkos::Bitset<DevSpace> mask(n_library);

    MutableTimeSeries prediction("prediction", n_prediction);

    for (int lib_size : lib_sizes) {
        SimplexLUT lut(n_prediction, E + 1);

        float rho_avg = 0.0f;

        for (int trial = 0; trial < samples; trial++) {
            mask_mirror.clear();

            for (int i = n_library - lib_size; i < n_library; i++) {
                int r = std::uniform_int_distribution<>(0, i)(engine);
                mask_mirror.set(mask_mirror.test(r) ? i : r);
            }

            Kokkos::deep_copy(mask, mask_mirror);

            Kokkos::parallel_for(
                "EDM::ccm::sample", n_prediction, KOKKOS_LAMBDA(int i) {
                    int selected = 0;

                    for (int j = 0; j < n_library && selected < E + 1; j++) {
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
                Kokkos::subview(target,
                                std::make_pair(shift, target.extent_int(0))),
                prediction);

            rho_avg += rho;
        }

        std::cout << "lib_size=" << lib_size
                  << " avg_rho=" << (rho_avg / samples) << std::endl;

        rhos.push_back(rho_avg / samples);
    }

    Kokkos::Profiling::popRegion();

    return rhos;
}

} // namespace edm
