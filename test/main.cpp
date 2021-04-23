#define DOCTEST_CONFIG_IMPLEMENT
#include <Kokkos_Core.hpp>
#include <doctest/doctest.h>

int main(int argc, char **argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    return doctest::Context(argc, argv).run();
}
